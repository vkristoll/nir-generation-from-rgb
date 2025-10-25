# models/discriminator.py

'''
References: 
1. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, ‘‘Image-to-image translation
with conditional adversarial networks,’’ in Proc. IEEE Conf. Comput. Vis.
Pattern Recognit. (CVPR), Jul. 2017, pp. 5967–5976.
2. https://www.codespeedy.com/image-to-image-translation-in-pytorch/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import dim_d, dim_in_ch

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

def conv_n(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
    )

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = conv(dim_in_ch, dim_d, 4, 2, 1)
        self.c2 = conv_n(dim_d, dim_d*2, 4, 2, 1)
        self.c3 = conv_n(dim_d*2, dim_d*4, 4, 2, 1)
        self.c4 = conv_n(dim_d*4, dim_d*8, 4, 1, 1)
        self.c5 = conv(dim_d*8, 1, 4, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        xy = F.leaky_relu(self.c1(xy), 0.2)
        xy = F.leaky_relu(self.c2(xy), 0.2)
        xy = F.leaky_relu(self.c3(xy), 0.2)
        xy = F.leaky_relu(self.c4(xy), 0.2)
        xy = self.c5(xy)
        return self.sigmoid(xy)