import torch
from torch import nn

def get_loss(config):
    loss_fn = {}
    for t in config['tasks']:
        loss_fn[t] = nn.CrossEntropyLoss()
    return loss_fn