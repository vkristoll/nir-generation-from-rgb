# src/utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

def save_visuals(im3b_t, realNIR_t, predim, epoch, i, log_path):
    """
    Saves a 2x2 grid of input, generated, ground truth, and composite output images.

    """
    A = im3b_t[0].detach().cpu()         # shape: [3, H, W]
    realNIR = realNIR_t[0].detach().cpu()  # shape: [1, H, W]

    # Compose RGB-style visuals
    B3 = torch.stack([realNIR.squeeze(0), A[2], A[1]])  # ground truth + red + green
    B4 = torch.stack([predim.squeeze(0), A[2], A[1]])   # generated + red + green

    plt.figure(figsize=(10, 10))
    titles = ["input image234", "generated image", "ground truth", "output image234"]
    images = [
        B3,
        predim.unsqueeze(0),     # shape [1, 1, H, W]
        realNIR.unsqueeze(0),    # shape [1, 1, H, W]
        B4
    ]

    for idx, img in enumerate(images):
        plt.subplot(2, 2, idx + 1)
        plt.axis("off")
        plt.title(titles[idx])
        plt.imshow(np.transpose(vutils.make_grid(img, nrow=1, padding=5, normalize=True), (1, 2, 0)))

    filename = f"pix2pix-{epoch}_{i+1}.png"
    plt.savefig(os.path.join(log_path, filename))
    plt.close()