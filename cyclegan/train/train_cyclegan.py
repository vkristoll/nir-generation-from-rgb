# train/train_cyclegan.py

import os
import time
import torch
import itertools
from models.generator import GeneratorResNet
from models.discriminator import Discriminator
from models.blocks import weights_init_normal
from data.preprocess import load_and_pad_image
from data.loader import generate_patches, BGRtoRGB
from utils.losses import get_loss_functions
from utils.visualizer import plot_cycle_images
from utils.metrics import track_losses, export_metrics_csv

def train_cycle_gan(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and pad images
    imA, _ = load_and_pad_image(os.path.join(config["data_dir_A"], config["image_A"]), config["bandsRGB"], config["patch_size"])
    imB, _ = load_and_pad_image(os.path.join(config["data_dir_B"], config["image_B"]), config["bandsRGB"], config["patch_size"])

    # Initialize models
    input_shape = (config["bandsRGB"], config["patch_size"], config["patch_size"])
    G_AB = GeneratorResNet(input_shape, config["n_residual_blocks"]).to(device)
    G_BA = GeneratorResNet(input_shape, config["n_residual_blocks"]).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Loss functions
    criterion_GAN, criterion_cycle, criterion_identity = get_loss_functions()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=config["lr"], betas=(config["b1"], config["b2"]))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))

    # LR schedulers
    lambda_lr = lambda epoch: 1.0 - max(0, epoch + config["epoch_offset"] - config["decay_epoch"]) / (config["n_epochs"] - config["decay_epoch"])
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_lr)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_lr)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_lr)

    # Tracking
    loss_dict = {k: [] for k in ["id_A", "id_B", "identity", "GAN_AB", "GAN_BA", "GAN", "cycle_A", "cycle_B", "cycle", "G", "realA", "fakeA", "D_A", "realB", "fakeB", "D_B", "D"]}
    lr_dict = {k: [] for k in ["G", "D_A", "D_B"]}
    time_list = []

    train_steps = (imA.shape[1] // config["patch_size"]) * (imA.shape[2] // config["patch_size"])
    start_time = time.time()

    for epoch in range(config["n_epochs"]):
        epoch_losses = {k: 0 for k in loss_dict}

        for step in range(train_steps):
            real_A = torch.tensor(generate_patches(imA, config["batch_size"], config["patch_size"])).to(device)
            real_B = torch.tensor(generate_patches(imB, config["batch_size"], config["patch_size"])).to(device)

            valid = torch.ones(real_A.size(0), *D_A.output_shape).to(device)
            fake = torch.zeros_like(valid)

            # Train Generators
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator A
            optimizer_D_A.zero_grad()
            loss_realA = criterion_GAN(D_A(real_A), valid)
            loss_fakeA = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_realA + loss_fakeA) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # Train Discriminator B
            optimizer_D_B.zero_grad()
            loss_realB = criterion_GAN(D_B(real_B), valid)
            loss_fakeB = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_realB + loss_fakeB) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # Accumulate losses
            for k, v in zip(epoch_losses.keys(), [loss_id_A, loss_id_B, loss_identity, loss_GAN_AB, loss_GAN_BA, loss_GAN,
                                                  loss_cycle_A, loss_cycle_B, loss_cycle, loss_G, loss_realA, loss_fakeA,
                                                  loss_D_A, loss_realB, loss_fakeB, loss_D_B, loss_D]):
                epoch_losses[k] += v.item()

        # Average losses
        loss_avgs = {k: epoch_losses[k] / train_steps for k in epoch_losses}
        track_losses(loss_dict, loss_avgs)

        # Step LR schedulers
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Track learning rates and time
        lr_dict["G"].append(optimizer_G.param_groups[0]['lr'])
        lr_dict["D_A"].append(optimizer_D_A.param_groups[0]['lr'])
        lr_dict["D_B"].append(optimizer_D_B.param_groups[0]['lr'])
        time_list.append(time.time() - start_time)

        # Logging
        print(f"Epoch [{epoch+1}/{config['n_epochs']}]: loss_G = {loss_avgs['G']:.6f}, loss_D = {loss_avgs['D']:.6f}")

        # Save sample images
        with torch.no_grad():
            G_AB.eval()
            G_BA.eval()
            plot_cycle_images(epoch+1, step+1,
                              real_A[0], real_B[0],
                              G_BA(real_A)[0], G_AB(real_B)[0],
                              fake_B[0], fake_A[0],
                              recov_A[0], recov_B[0],
                              config["log_path"], BGRtoRGB)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'G_AB.state_dict': G_AB.state_dict(),
            'G_BA.state_dict': G_BA.state_dict(),
            'D_A.state_dict': D_A.state_dict(),
            'D_B.state_dict': D_B.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        }, os.path.join(config["checkpoint_dir"], f"CycleGAN_epoch_{epoch+1}.pth"))

        # Export metrics
        export_metrics_csv(loss_dict, lr_dict, time_list, config["output_dir"], config["metrics_csv"])
