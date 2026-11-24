import torch
import random
import numpy as np

PNG_DIR = './png_data'
NORMAL_DIR = './png_data/normal'
ABNORMAL_DIR = './png_data/abnormal'
CHECKPOINT_DIR = './checkpoints'


#IMAGE
IMG_SIZE = 256
IMG_TUPLE = (IMG_SIZE, IMG_SIZE)

#Model
LATENT_DIM = 32

#Training
BATCH_SIZE = 16
EPOCHS = 1000
LR = 1e-4
SEED = 42
SPLITS = (0.7, 0.15, 0.15)


# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def setup_reproducibility():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
