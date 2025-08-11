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

from utils import _lr_scheduler, accuracy
from test import validation
from gcn import GCN, GCN_Graph

if __name__ == "__main__":
    # Example usage:
    # Suppose:
    # B = 4, T = 5, R = 10, C = 16, hidden_channels = 32
    B, T, R, C = 4, 20, 56, 56
    hidden_channels = 64

    # Random input tensor with shape [B, T, R, C]
    x = torch.randn(B, T, R, C)

    # Define an example edge_index for a graph with R nodes.
    # Here we create a fully-connected directed graph (including self-loops)
    with open('./config/asl.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    # Get sequences for 3 words
    graphs_path = os.path.join(config.path, 'graphs')
    #'/home/tkg5kq/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/graphs/'
    # label_map = {'abdomen': 0, 'good': 1, 'yes': 2}
    classes, label_map = wlasl_dataloader.load_classes(config.config_file)
    # classes = ['abdomen', 'good', 'yes', 'disconnect', 'eye', 'government', 'interpreter', 'mature', 'ring', 'slave', 'sweet', 'twin']
    sequence, labels = wlasl_dataloader.collect_graph_data(classes, graphs_path, config, label_map)

    # Get random sequence for the data gathered
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    random_sequence = random.sample(range(len(sequence)), int(len(sequence)*.8))
    for i in range(len(sequence)):
        if i in random_sequence:
            X_train.append(sequence[i])
            y_train.append(labels[i])
        else:
            X_test.append(sequence[i])
            y_test.append(labels[i])

    # only first frames
    # train_loader = DataLoader([x[10] for x in X_train], batch_size=32, shuffle=True) 
    train_loader = DataLoader(X_train, batch_size=config.batch_size, shuffle=True) 
    test_loader = DataLoader(X_test, batch_size=config.batch_size, shuffle=False)
    
    # Create the model and forward pass.
    NUM_JOINTS = 56
    model = GCN_Graph(input_dim=3, 
                      hidden_dim=hidden_channels, 
                      output_dim=len(classes), 
                      num_layers=3, 
                      dropout=.5, 
                      parallel=config.parallel,
                      num_gpus=config.gpus,
                      num_frames=config.sample_num,
                      num_joints=NUM_JOINTS,
                      batch_size=config.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from torch_geometric.nn import DataParallel
    parallel = config.parallel
    if parallel:
        model = DataParallel(model)
        model = model.to(device)

    optimizer = SGD(model.parameters(), lr=0.08)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = _lr_scheduler(config, optimizer)


    for epoch in range(config.epochs):
        model.train()
        running_loss = 0
        total_runs = 0
        total_correct = 0
        if config.tqdm:
            loop = tqdm(train_loader, total=len(train_loader))
        else:
            loop = train_loader
        for i, batch in enumerate(loop): 
            if config.solver.type != 'monitor':
                if (i+1) == 1 or (i+1) % 10 == 0:
                    lr_scheduler.step(epoch + i / len(train_loader))
            optimizer.zero_grad()
            
            # Batch is NxVxTxC
            out = model(batch)
            gnd = batch[0].y
            if parallel:    
                gnd = gnd.to(device)
            loss = criterion(out, gnd)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            correct, total = accuracy(out.argmax(1), gnd)
            total_correct += correct
            total_runs += total 

        if (epoch+1) % config.freq == 0:
            print(f'Epoch {epoch+1}/{config.epochs}: Loss: {running_loss/total_runs}, Accuracy: {total_correct}/{total_runs} = {(total_correct*100)/total_runs:.2f}%, LR: {lr_scheduler.get_lr()[0]:.5f}')
            validation(model, test_loader)