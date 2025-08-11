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

# Function to convert edge list to adjacency matrix
# https://youcademy.org/conversion-between-edge-list-adjacency-matrix/
def convert_to_adjacency_matrix(edge_list, num_vertices):
  # Initialize the adjacency matrix with all zeros
  adj_mat = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
  
  # Iterate through all the edges
  source_list = edge_list[0]
  target_list = edge_list[1]
  for source, destination in zip(source_list, target_list):
    # Mark the edge between source and destination in adjacency matrix
    # torso, legs, left hand, right hand
    # 11-14 -> 0-3, 23-32 -> 4-13, 33-53 -> 14-34, 54-74 -> 35-55

    if source >= 11 and source <= 14:
        source = source - 11
        destination = destination - 11
    elif source >= 23 and source <= 74:
        source = source - 19
        destination = destination - 19
    
    
    adj_mat[source][destination] = 1

    #  Mark the edge between destination and source in adjacency matrix
    # (since the graph is bidirectional)
    adj_mat[destination][source] = 1

  
  return adj_mat

# Function to convert adjacency matrix to edge list
def convert_to_edge_list(adjacency_matrix):
  # Initialize an empty list to store the edge list
  edge_list = []

  # Iterate through the rows of the adjacency matrix
  for source in range(len(adjacency_matrix)):
    # Iterate through the columns of the adjacency matrix
    for destination in range(len(adjacency_matrix[source])):
      # If the value in the adjacency matrix is 1
      # Add an edge to the edge list from source to destinatoin
      if adjacency_matrix[source][destination] == 1:
        edge_list.append((source, destination))

  # Return the edge list
  return edge_list

def _lr_scheduler(config, optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler

def accuracy(out, y):
    correct = 0
    total = 0
    for pred, label in zip(out, y):
        total += 1
        correct += (pred == label)
    return correct, total