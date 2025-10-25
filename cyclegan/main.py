# main.py

import argparse
from config import config
from train.train_cyclegan import train_cycle_gan

def parse_args():
    parser = argparse.ArgumentParser(description="Train CycleGAN for RGB domain adaptation")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--patch_size", type=int, help="Patch size for image crops")
    parser.add_argument("--lr", type=float, help="Learning rate for optimizers")
    parser.add_argument("--res_blocks", type=int, help="Number of residual blocks in generator")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Override config with CLI arguments if provided
    if args.epochs is not None:
        config["n_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.patch_size is not None:
        config["patch_size"] = args.patch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.res_blocks is not None:
        config["n_residual_blocks"] = args.res_blocks

    train_cycle_gan(config)