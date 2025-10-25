# utils/losses.py

import torch
import torch.nn as nn

def get_loss_functions():
    """
    Returns the three main loss functions used in CycleGAN:
    - GAN loss (MSE)
    - Cycle consistency loss (L1)
    - Identity loss (L1)
    """
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    return criterion_GAN, criterion_cycle, criterion_identity