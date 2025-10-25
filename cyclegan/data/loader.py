# data/loader.py

import numpy as np
import torch
import random

def BGRtoRGB(BGR):
    """
    Converts a BGR patch to RGB format.
    """
    RGB = torch.zeros_like(BGR)
    RGB[0, :, :] = BGR[2, :, :]
    RGB[1, :, :] = BGR[1, :, :]
    RGB[2, :, :] = BGR[0, :, :]
    return RGB

def generate_patches(image, batch_size, patch_size):
    """
    Generates a batch of non-zero patches from a padded image.
    """
    bands, rows, cols = image.shape
    X = np.zeros((batch_size, bands, patch_size, patch_size), dtype=np.float32)

    i = 0
    c = 0
    while i < batch_size:
        row = random.randint(0, rows - patch_size)
        col = random.randint(0, cols - patch_size)
        patch = image[:, row:row + patch_size, col:col + patch_size]

        if np.mean(patch) != 0:
            X[c] = patch
            c += 1
            i += 1

    return X
