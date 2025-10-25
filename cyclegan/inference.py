# inference.py

import argparse
import numpy as np
import torch
import tifffile as tiff
from models.generator import GeneratorResNet
from data.preprocess import load_and_pad_image

def parse_args():
    parser = argparse.ArgumentParser(description="CycleGAN inference on multispectral TIFF image")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input TIFF image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained CycleGAN checkpoint (.pth)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output TIFF")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for inference")
    parser.add_argument("--bands", type=int, default=3, help="Number of input channels (e.g. RGB)")
    parser.add_argument("--res_blocks", type=int, default=3, help="Number of residual blocks in generator")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    return parser.parse_args()

def run_inference(args):
    device = torch.device(args.device)
    input_shape = (args.bands, args.patch_size, args.patch_size)

    # Load model
    model = GeneratorResNet(input_shape, args.res_blocks).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['G_BA.state_dict'])
    model.eval()

    # Load and pad image
    padded, rows, cols = load_and_pad_image(args.input_image, args.bands, args.patch_size)
    padded_rows, padded_cols = padded.shape[1], padded.shape[2]

    # Inference
    output = np.zeros((args.bands, padded_rows, padded_cols), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, padded_rows, args.patch_size):
            for j in range(0, padded_cols, args.patch_size):
                patch = padded[:, i:i+args.patch_size, j:j+args.patch_size]
                patch_tensor = torch.tensor(patch).unsqueeze(0).to(device)
                pred = model(patch_tensor).squeeze(0).cpu().numpy()
                output[:, i:i+args.patch_size, j:j+args.patch_size] = pred

    # Crop to original size
    final_output = output[:, :rows, :cols]
    tiff.imwrite(args.output_path, final_output)
    print(f"Saved output to {args.output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)