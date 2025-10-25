# data/preprocess.py

import numpy as np
import math
import tifffile as tiff

def load_and_pad_image(filepath, bands, patch_size):
    """
    Loads a TIFF image, transposes it to (C, H, W), and pads it to be divisible by patch_size.
    Returns the padded image and its original shape.
    """
    im = tiff.imread(filepath).transpose(2, 0, 1)
    rows, cols = im.shape[1], im.shape[2]

    impad_rows = rows if rows % patch_size == 0 else (rows // patch_size + 1) * patch_size
    impad_cols = cols if cols % patch_size == 0 else (cols // patch_size + 1) * patch_size

    impad = np.zeros((bands, impad_rows, impad_cols), dtype=np.float32)
    impad[:, 0:rows, 0:cols] = im[0:bands, :, :]

    return impad, (rows, cols)