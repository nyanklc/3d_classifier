import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data import get_processed_dataset, split_dataset, visualize_mat
from model import Generator, Discriminator, ResNet50, DummyGenerator

PITCH_NR = 88
TIME_LEN = 5000
LATENT_DIM = 128
BATCH_SIZE = 16
NR_EPOCHS = 1
TRAIN_RATIO = 0.1
OUT_DIR = "./out/"

def main():
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
