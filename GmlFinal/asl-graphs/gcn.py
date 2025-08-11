import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import sys
import wlasl_dataloader
import yaml
from dotmap import DotMap
import torch

# from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import SGD
from torchvision.transforms.functional import pad
from torch_geometric.data import InMemoryDataset

from sklearn.model_selection import train_test_split
import random
from torch_geometric.loader import DataLoader

import torch.optim as optim
from lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from tqdm import tqdm 
from torch_geometric.utils import unbatch_edge_index, unbatch

from utils import *
from test import validation

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs,
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        convs = []
        # A list of 1D batch normalization layers
        bns = []
        convs.append(GCNConv(input_dim, hidden_dim))
        bns.append(nn.BatchNorm1d(hidden_dim))
        self.more_layers = True
        if self.more_layers:
            convs.append(GCNConv(hidden_dim, hidden_dim*2))
            bns.append(nn.BatchNorm1d(hidden_dim*2))
            # convs.append(GCNConv(hidden_dim*2, hidden_dim*2))
            # bns.append(nn.BatchNorm1d(hidden_dim*2))
            convs.append(GCNConv(hidden_dim*2, hidden_dim))
            bns.append(nn.BatchNorm1d(hidden_dim))
        convs.append(GCNConv(hidden_dim, output_dim))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        num_layers = len(self.convs)

        # The log softmax layer
        self.softmax = nn.LogSoftmax()

        # Probability of an element getting zeroed
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=self.dropout)

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.
        for i in range(len(self.convs)-1):
            # print(self.convs[i])
            # print(x.shape, adj_t.shape)
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = self.relu(x)
            x = self.drop(x)

        if self.return_embeds:
            # x = self.softmax(x)
            return x
        else:
            x = self.convs[-1](x, adj_t)
            return x

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim, 
                output_dim, 
                num_layers, 
                dropout, 
                parallel,
                num_gpus=4,
                num_frames=20,
                num_joints=56,
                batch_size=32):
        super(GCN_Graph, self).__init__()

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = global_mean_pool

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # Output layer
        # self.linear = nn.Linear(hidden_dim, output_dim)
        self.layers = True
        if self.layers:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.parallel = parallel

        self.batch_size = batch_size
        self.joint_size = num_joints
        self.frame_size = num_frames
        self.num_gpus = num_gpus
        self.hidden_dim = hidden_dim

    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):

        if not self.parallel:
            output = torch.empty(self.frame_size, ((len(batched_data[0].x)*len(batched_data))//self.frame_size), self.hidden_dim) # T (N*V) C
            for i, data in enumerate(batched_data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x = self.gnn_node(x, edge_index) # (N*V)xC for one single frame
                output[i, :, :] = x
            # Resulting output is Tx(N*V)xC
            # TODO change this rudementary mean of the temporal dimension to attention in time or LSTM (use the one from ActionCLIP for simplicity)
            x = output.permute(1, 0, 2).mean(1) # Tx(N*V)xC -> (N*V)xTxC -> (N*V)xC
            x = self.pool(x, batch=batch) # Mean pool results in NxC
        else:
            x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
            device = x.device

            batch_desc = torch.repeat_interleave(torch.arange(self.frame_size), (len(x)//(56*self.frame_size))*56).to(device=device)
            unbatched_x = unbatch(x, batch_desc)
            unbatched_edges = unbatch_edge_index(edge_index, batch_desc)

            # raise ValueError(unbatched_x.view(8, -1, 56, 3).permute(0, 2, 1, 3).shape)
            output = torch.empty(self.frame_size, (len(x)//(56*self.frame_size))*self.joint_size, self.hidden_dim) # T (N*V) C
            # output= torch.stack([self.gnn_node(new_x.to(x.device), new_edge_index.to(x.device)) for new_x, new_edge_index in zip(unbatched_x, unbatched_edges)])
            for i, new_x in enumerate(unbatched_x):
                # raise ValueError(new_x.shape, len(unbatched_x), len(unbatched_edges))
                x = self.gnn_node(new_x.to(device), unbatched_edges[i].to(device))
                output[i, :, :] = x
            
            x = output.permute(1, 0, 2).mean(1) # Tx(N*V)xC -> (N*V)xTxC -> (N*V)xC
            batch_desc = torch.repeat_interleave(torch.arange(len(x)//(self.joint_size)), self.joint_size).to(device=device)
            x = x.to(device)
            x = self.pool(x, batch=batch_desc) # Mean pool results in NxC

        # print(f'Pool {x.shape}')
        if self.layers:
            x = self.relu(self.fc1(x)) # Nx64
            x = self.relu(self.fc2(x)) # Nx32
            x = self.fc3(x) # NxL
        else:
            x = self.fc3(x)
        # torch.stack(output)
        return x