# src/data_loader.py
import numpy as np
import math
import tifffile as tiff
from config import patch_size, bands, data_path

def load_and_pad_image():
    im = tiff.imread(data_path).transpose(2, 0, 1)
    rows, cols = im.shape[1], im.shape[2]
    impad_rows = math.ceil(rows / patch_size) * patch_size
    impad_cols = math.ceil(cols / patch_size) * patch_size
    impad = np.zeros((bands, impad_rows, impad_cols), dtype=np.float32)
    impad[:, :rows, :cols] = im
    return impad, rows, cols

def generate_3b(impad, row_idx, col_idx, batch_size, gen_in_ch):
    X = np.zeros((batch_size, gen_in_ch, patch_size, patch_size), dtype=np.float32)
    for i in range(batch_size):
        X[i] = impad[0:gen_in_ch, row_idx[i]:row_idx[i]+patch_size, col_idx[i]:col_idx[i]+patch_size]
    return X

def generate_real_NIR(impad, row_idx, col_idx, batch_size):
    X = np.zeros((batch_size, 1, patch_size, patch_size), dtype=np.float32)
    for i in range(batch_size):
        X[i] = impad[3, row_idx[i]:row_idx[i]+patch_size, col_idx[i]:col_idx[i]+patch_size]
    return X