import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy
import os

from data import VGDataset, plot_voxel_grid, find_small_scale_vgs, get_label_str, modelnet40_label_to_idx
from model import Classifier3D, AutoEncoder3D, Classifier3DWithEncoder

OUTPUT_DIR = "./out/"

FILTER = "monster"

filenames = []
list_losses_train = []
list_losses_val = []
list_accuracies_train = []
list_accuracies_val = []
list_accuracy_test = []
list_accuracies_test = []
list_BATCH_SIZE = []

for root, _, files in os.walk(OUTPUT_DIR):
    for file in files:
        if not file.endswith(".pth"): continue

        if not FILTER in file: continue

        try:
            print(f"{os.path.join(root, file)}")
            checkpoint = torch.load(os.path.join(root, file), weights_only=False)
            filenames.append(file)
            # optional fields
            list_losses_train.append(checkpoint.get("losses_train", []))
            list_losses_val.append(checkpoint.get("losses_val", []))
            list_accuracies_train.append(checkpoint.get("accuracies_train", []))
            list_accuracies_val.append(checkpoint.get("accuracies_val", []))
            list_accuracy_test.append(checkpoint.get("accuracy_test", []))
            list_accuracies_test.append(checkpoint.get("accuracies_test", []))

            BATCH_SIZE = 0
            if "BATCH_SIZE" in checkpoint:
                BATCH_SIZE = checkpoint["BATCH_SIZE"]
            list_BATCH_SIZE.append(BATCH_SIZE)
        except:
            print("FAIL")

print(f"hello {len(list_losses_train)}")
saveto = f"{FILTER}/" if FILTER != "" else "./"
Path(OUTPUT_DIR + "plots/" + saveto).mkdir(parents=True, exist_ok=True)
for i in range(len(list_losses_train)):
    # loss batch
    plt.figure(figsize=(15,8))
    plt.plot(list_losses_train[i], label="Training Loss")
    # plt.plot(list_losses_val[i], label="Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    tacc = "" if len(list_accuracy_test[i])==0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = {"" if len(list_accuracies_test[i])==0 else list_accuracies_test[i][-1]}
    plt.title(f"Loss (per batch), Test Accuracy {tacc}")
    plt.grid()
    plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch.png", dpi=300, bbox_inches="tight")
    plt.close()


    # loss batch
    plt.figure(figsize=(15,8))
    # plt.plot(list_losses_train[i], label="Training Loss")
    plt.plot(list_losses_val[i], label="Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    tacc = "" if len(list_accuracy_test[i])==0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = {"" if len(list_accuracies_test[i])==0 else list_accuracies_test[i][-1]}
    # {filenames[i]} --
    plt.title(f"Loss (per batch), Test Accuracy {tacc}")
    plt.grid("minor")
    plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch.png", dpi=300, bbox_inches="tight")
    plt.close()

    # # loss epoch
    # plt.figure(figsize=(10,5))
    # plt.plot(train_loss_avg_per_epoch, label="Training Loss")
    # plt.plot(val_loss_avg_per_epoch, label="Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title(f"Loss (per epoch), Test Accuracy {accuracy_test[-1]}")
    # plt.grid()
    # plt.show()

    # accuracy
    plt.figure(figsize=(15,8))
    plt.plot(list_accuracies_train[i], label="Training Accuracy")
    plt.plot(list_accuracies_val[i], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    tacc = "" if len(list_accuracy_test[i])==0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = {"" if len(list_accuracies_test[i])==0 else list_accuracies_test[i][-1]}
    plt.title(f"Accuracy, Test Accuracy {tacc}")
    plt.grid("minor")
    plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
