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

from config import patch_size, in_ch
from data_loader import load_and_pad_image, generate_3b, generate_real_NIR
from model.AWAN_model import AWAN
from utils import save_visuals

def parse_args():
    parser = argparse.ArgumentParser(description="Train AWAN model")
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

    model = AWAN().to(device)
        
    L1 = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),eps=1e-08, weight_decay=0)
   
    L1_losses = []
    time_list = []
   
    train_steps=350
    start_time = time.time()
    for epoch in range(args.epochs):
        L1_loss_total = 0
   
        for i in range(train_steps):
            row_idx = random.sample(range(0, impad.shape[1] - patch_size), args.batch_size)
            col_idx = random.sample(range(0, impad.shape[2] - patch_size), args.batch_size)

            im3b = torch.tensor(generate_3b(impad, row_idx, col_idx, args.batch_size, in_ch)).to(device)
            realNIR = torch.tensor(generate_real_NIR(impad, row_idx, col_idx, args.batch_size)).to(device)

            # Train model
            NIRpred=model(im3b)
            model.zero_grad()                   
            L1_loss = L1(NIRpred,realNIR)                         
            L1_loss.backward()        
            optimizer.step()
            
            # Track loss
            L1_loss_total+=L1_loss.item()

        # Epoch averages
        L1_losses.append(L1_loss_total / train_steps)
        time_list.append(time.time() - start_time)
        
        # Logging
        print(f"Epoch [{epoch+1}/{args.epochs}]  L1: {L1_losses[-1]:.6f}")   
        print(f"--- {time_list[-1]:.2f} seconds ---")

        # Save visuals
        with torch.no_grad():
            model.eval()
            predim = model(im3b)[0].detach().cpu()
            model.train()
        save_visuals(im3b, realNIR, predim, epoch+1, i, args.log_path)

        # Save models
        torch.save(model.state_dict(), os.path.join(args.model_path, f"awan_{epoch+1}.pth"))
       
    # Save metrics
    os.makedirs("outputs", exist_ok=True)
    metrics = np.array(list(zip(L1_losses, time_list)))
    np.savetxt("outputs/metrics_pix2pix.csv", metrics, delimiter=',', header="L1,time", fmt='%10.6f')

if __name__ == "__main__":
    args = parse_args()
    main(args)