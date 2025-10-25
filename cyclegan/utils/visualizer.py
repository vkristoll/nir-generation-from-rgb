# utils/visualizer.py

import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch

def plot_cycle_images(epoch, step, real_A, real_B, idBA, idAB, fB, fA, rA, rB, log_path, BGRtoRGB):
    """
    Plots and saves a grid of CycleGAN images for visual inspection.
    """
    plt.figure(figsize=(10, 10))

    images = [
        (real_A, "realA"),
        (real_B, "realB"),
        (idBA, "idBA"),
        (idAB, "idAB"),
        (fB, "fB"),
        (fA, "fA"),
        (rA, "rA"),
        (rB, "rB")
    ]

    for idx, (img, title) in enumerate(images, start=1):
        plt.subplot(4, 2, idx)
        plt.axis("off")
        plt.title(title)
        rgb = BGRtoRGB(img.cpu())
        grid = vutils.make_grid(rgb, nrow=4, padding=5, normalize=True)
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))

    filename = f"Cyclegan-{epoch}_{step}.png"
    plt.savefig(os.path.join(log_path, filename))
    plt.close()