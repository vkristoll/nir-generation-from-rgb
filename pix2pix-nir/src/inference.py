# src/inference.py

import os
import math
import argparse
import numpy as np
import torch
import tifffile as tiff

from data_loader import load_and_pad_image
from models.generator import Generator

def run_inference(args):
    device = torch.device(args.device)

    # Load model
    GenModel = Generator().to(device)
    GenModel.load_state_dict(torch.load(args.model_path, map_location=device))
    GenModel.eval()

    # Load and pad image
    impad, rows, cols = load_and_pad_image(args.input_image, args.patch_size)

    padded_rows, padded_cols = impad.shape[1], impad.shape[2]    
    output = np.zeros((1, padded_rows, padded_cols), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, padded_rows, args.patch_size):
            patches = []
            for j in range(0, padded_cols, args.patch_size):
                patch = impad[:, i:i+args.patch_size, j:j+args.patch_size]
                patches.append(patch)
            batch = torch.tensor(patches, dtype=torch.float32).to(device)
            predictions = GenModel(batch).detach().cpu().numpy()
            for j, pred in enumerate(predictions):
                row = i // args.patch_size
                col = j
                output[:, row*args.patch_size:(row+1)*args.patch_size, col*args.patch_size:(col+1)*args.patch_size] = pred

    final_output = output[:, :rows, :cols]
    tiff.imwrite(args.output_image, final_output)
    print(f"Saved output to {args.output_image}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference on multispectral image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained generator model (.pth)")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input TIFF image")
    parser.add_argument("--output_image", type=str, default="output_NIR.tiff", help="Path to save output TIFF")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)