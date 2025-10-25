# config.py

config = {
    # Data
    "patch_size": 256,
    "bandsRGB": 3,
    "batch_size": 14,

    # Training
    "n_epochs": 80,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "decay_epoch": 70,
    "epoch_offset": 1,
    "n_residual_blocks": 3,

    # Paths
    "data_dir_A": "path to 1st date image",
    "data_dir_B": "path to 2nd date image",
    "image_A": "A.tiff",
    "image_B": "B.tiff",
    "log_path": " ",
    "checkpoint_dir": " ",
    "metrics_csv": " ",
    "output_dir": " "
}