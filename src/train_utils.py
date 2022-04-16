import copy
import os
import math

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR

def create_optimizer(config, model_params):
    """Create optimizer for training process"""
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model_params, lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=config['lr'], betas=(0.9, 0.999))
    elif config['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(model_params, lr=config['lr'])
    else:
        assert False, "Unknown optimizer type"

    return optimizer

def create_lr_scheduler(optimizer, config):
    """Create learning rate scheduler for training process"""
    if config['lr_type'] == 'step_lr':
        lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    elif config['lr_type'] == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    elif config['lr_type'] == 'cosin':
        lf = lambda x: (((1 + math.cos(x * math.pi / config['NUM_EPOCH'])) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        # lr_scheduler.last_epoch = params['start_epoch'] - 1
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)
    else:
        raise TypeError

    return lr_scheduler