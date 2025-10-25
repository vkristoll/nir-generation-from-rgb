# models/generator.py
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
from src.config import gen_in_ch, gen_out_ch, dim_g

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

def conv_n(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
    )

def tconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)

def tconv_n(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
    )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n1 = conv(gen_in_ch, dim_g, 4, 2, 1)
        self.n2 = conv_n(dim_g, dim_g*2, 4, 2, 1)
        self.n3 = conv_n(dim_g*2, dim_g*4, 4, 2, 1)
        self.n4 = conv_n(dim_g*4, dim_g*8, 4, 2, 1)
        self.n5 = conv_n(dim_g*8, dim_g*8, 4, 2, 1)
        self.n6 = conv_n(dim_g*8, dim_g*8, 4, 2, 1)
        self.n7 = conv_n(dim_g*8, dim_g*8, 4, 2, 1)
        self.n8 = conv(dim_g*8, dim_g*8, 4, 2, 1)

        self.m1 = tconv_n(dim_g*8, dim_g*8, 4, 2, 1)
        self.m2 = tconv_n(dim_g*8*2, dim_g*8, 4, 2, 1)
        self.m3 = tconv_n(dim_g*8*2, dim_g*8, 4, 2, 1)
        self.m4 = tconv_n(dim_g*8*2, dim_g*8, 4, 2, 1)
        self.m5 = tconv_n(dim_g*8*2, dim_g*4, 4, 2, 1)
        self.m6 = tconv_n(dim_g*4*2, dim_g*2, 4, 2, 1)
        self.m7 = tconv_n(dim_g*2*2, dim_g*1, 4, 2, 1)
        self.m8 = tconv(dim_g*1*2, gen_out_ch, 4, 2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n1 = self.n1(x)
        n2 = self.n2(F.leaky_relu(n1, 0.2))
        n3 = self.n3(F.leaky_relu(n2, 0.2))
        n4 = self.n4(F.leaky_relu(n3, 0.2))
        n5 = self.n5(F.leaky_relu(n4, 0.2))
        n6 = self.n6(F.leaky_relu(n5, 0.2))
        n7 = self.n7(F.leaky_relu(n6, 0.2))
        n8 = self.n8(F.leaky_relu(n7, 0.2))

        m1 = torch.cat([F.dropout(self.m1(F.relu(n8)), 0.5, training=True), n7], 1)
        m2 = torch.cat([F.dropout(self.m2(F.relu(m1)), 0.5, training=True), n6], 1)
        m3 = torch.cat([F.dropout(self.m3(F.relu(m2)), 0.5, training=True), n5], 1)
        m4 = torch.cat([self.m4(F.relu(m3)), n4], 1)
        m5 = torch.cat([self.m5(F.relu(m4)), n3], 1)
        m6 = torch.cat([self.m6(F.relu(m5)), n2], 1)
        m7 = torch.cat([self.m7(F.relu(m6)), n1], 1)
        m8 = self.m8(F.relu(m7))

        return self.sigmoid(m8)
