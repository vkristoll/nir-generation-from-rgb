# models/discriminator.py

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, height, width = input_shape
        self.output_shape = (1, height // 8, width // 8)  # PatchGAN output shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, normalize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)