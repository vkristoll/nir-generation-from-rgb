# src/config.py
patch_size = 256
bands = 4
gen_in_ch = 3
gen_out_ch = 1
dim_g = 16
dim_d = 16
batch_size = 32
epochs = 50
L1_lambda = 100

device = "cuda:0"
data_path = "../data/train_image.tiff"
log_path = "../logs/"
model_path = "../models/"