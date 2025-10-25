# src/train.py
import os
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import patch_size, gen_in_ch, L1_lambda
from data_loader import load_and_pad_image, generate_3b, generate_real_NIR
from models.generator import Generator
from models.discriminator import Discriminator
from utils import save_visuals

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pix2Pix model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument("--model_path", type=str, default="models", help="Path to save models")
    parser.add_argument("--log_path", type=str, default="logs", help="Path to save visualizations")
    return parser.parse_args()

def main(args):
    device = torch.device(args.device)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    impad, rows, cols = load_and_pad_image()
    train_steps = int(math.floor(cols / patch_size) * (rows / patch_size))

    Gen = Generator().to(device)
    Disc = Discriminator().to(device)

    BCE = nn.BCELoss()
    L1 = nn.L1Loss()

    Gen_optim = optim.Adam(Gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    Disc_optim = optim.Adam(Disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    Disc_losses, Gen_losses, L1_losses = [], [], []
    r_gan_losses, f1_gan_losses, f2_gan_losses, time_list = [], [], [], []

    start_time = time.time()
    for epoch in range(args.epochs):
        Disc_loss_total = Gen_loss_total = L1_loss_total = 0
        r_gan_loss_total = f1_gan_loss_total = f2_gan_loss_total = 0

        for i in range(train_steps):
            row_idx = random.sample(range(0, impad.shape[1] - patch_size), args.batch_size)
            col_idx = random.sample(range(0, impad.shape[2] - patch_size), args.batch_size)

            r_masks = torch.ones(args.batch_size, 1, 30, 30).to(device)
            f_masks = torch.zeros(args.batch_size, 1, 30, 30).to(device)

            real3b = torch.tensor(generate_3b(impad, row_idx, col_idx, args.batch_size, gen_in_ch)).to(device)
            realNIR = torch.tensor(generate_real_NIR(impad, row_idx, col_idx, args.batch_size)).to(device)

            # Train Discriminator
            Disc.zero_grad()
            r_patch = Disc(real3b, realNIR)
            r_gan_loss = BCE(r_patch, r_masks)

            fake = Gen(real3b)
            f_patch = Disc(real3b, fake.detach())
            f1_gan_loss = BCE(f_patch, f_masks)

            Disc_loss = r_gan_loss + f1_gan_loss
            Disc_loss.backward()
            Disc_optim.step()

            # Train Generator
            Gen.zero_grad()
            f_patch = Disc(real3b, fake)
            f2_gan_loss = BCE(f_patch, r_masks)
            L1_loss_val = L1(fake, realNIR)
            Gen_loss = f2_gan_loss + L1_lambda * L1_loss_val
            Gen_loss.backward()
            Gen_optim.step()

            # Track losses
            Disc_loss_total += Disc_loss.item()
            Gen_loss_total += Gen_loss.item()
            L1_loss_total += L1_loss_val.item()
            r_gan_loss_total += r_gan_loss.item()
            f1_gan_loss_total += f1_gan_loss.item()
            f2_gan_loss_total += f2_gan_loss.item()

        # Epoch averages
        Disc_losses.append(Disc_loss_total / train_steps)
        Gen_losses.append(Gen_loss_total / train_steps)
        L1_losses.append(L1_loss_total / train_steps)
        r_gan_losses.append(r_gan_loss_total / train_steps)
        f1_gan_losses.append(f1_gan_loss_total / train_steps)
        f2_gan_losses.append(f2_gan_loss_total / train_steps)
        time_list.append(time.time() - start_time)

        # Logging
        print(f"Epoch [{epoch+1}/{args.epochs}]  Disc: {Disc_losses[-1]:.6f}  Gen: {Gen_losses[-1]:.6f}  L1: {L1_losses[-1]:.6f}")
        print(f"         r_gan: {r_gan_losses[-1]:.6f}  f1_gan: {f1_gan_losses[-1]:.6f}  f2_gan: {f2_gan_losses[-1]:.6f}")
        print(f"--- {time_list[-1]:.2f} seconds ---")

        # Save visuals
        with torch.no_grad():
            Gen.eval()
            fakeim = Gen(real3b)[0].detach().cpu()
            Gen.train()
        save_visuals(real3b, realNIR, fakeim, epoch+1, i, args.log_path)

        # Save models
        torch.save(Gen.state_dict(), os.path.join(args.model_path, f"Gen_{epoch+1}.pth"))
        torch.save(Disc.state_dict(), os.path.join(args.model_path, f"Disc_{epoch+1}.pth"))

    # Save metrics
    os.makedirs("outputs", exist_ok=True)
    metrics = np.array(list(zip(Disc_losses, Gen_losses, L1_losses, r_gan_losses, f1_gan_losses, f2_gan_losses, time_list)))
    np.savetxt("outputs/metrics_pix2pix.csv", metrics, delimiter=',', header="Disc,Gen,L1,r_gan,f1_gan,f2_gan,time", fmt='%10.6f')

if __name__ == "__main__":
    args = parse_args()
    main(args)