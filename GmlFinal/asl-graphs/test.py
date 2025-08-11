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

from utils import accuracy

def validation(model, test_loader, parallel=False):
    model.eval()

    total_correct = 0
    total_cnt = 0
    for i, batch in enumerate(test_loader):
        out = model(batch)
        pred = out.argmax(1)
        gnd = batch[0].y
        if parallel:
            gnd = gnd.to(device, dtype=int)

        correct, total = accuracy(pred, gnd)
        total_correct += correct
        total_cnt += total

    print(f'Validation Accuracy: {(total_correct*100)/total_cnt:.3f}%')