import os
from pathlib import Path

for root, _, files in os.walk("./data/out"):
    for file in files:
        if "rotated" in file:
            print(f"removing {os.path.join(root, file)}")
            os.remove(os.path.join(root, file))

            # out_path = os.path.join(root, file).replace("ModelNet40", "out_augmenteds_backup")
            # print(f"HOBA {root.replace("ModelNet40", "out_augmenteds_backup")}")
            # Path(root.replace("ModelNet40", "out_augmenteds_backup")).mkdir(parents=True, exist_ok=True)
            # print(f"olala {os.path.join(root, file)} ---> {out_path}")
            # input()
            # os.rename(os.path.join(root, file), out_path)

exit()










import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy

from data import VGDataset, plot_voxel_grid, find_small_scale_vgs, get_label_str
from model import Classifier3D, AutoEncoder3D, Classifier3DWithEncoder, AutoEncoderLabel, Classifier3DEncodedLabel

# DATASET_PATH = "./data/ModelNet40/"
DATASET_PROCESSED_PATH = "./data/out/"
VOXEL_SIZE = 0.02
MODEL_GRID_SHAPE_FILE = DATASET_PROCESSED_PATH + "model_grid_shape.txt"

BATCH_SIZE = 64
NR_EPOCHS = 30

VALIDATION_PERCENTAGE = 0.2

# train, val, and test datasets loaded, lenghts: 31464, 1967, 2465
TRAIN_LEN = 31464
VAL_LEN = 1967
TEST_LEN = 2465

OUTPUT_DIR = "./out/"

def plot(losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test):
    # loss batch
    plt.figure(figsize=(10,5))
    plt.plot(losses_train, label="Training Loss")
    plt.plot(losses_val, label="Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss (per batch), Test Accuracy {accuracy_test[-1]}")
    plt.grid()
    plt.show()

    # loss epoch
    train_batch_count_per_epoch: float = float(TRAIN_LEN) / float(BATCH_SIZE) / len(accuracies_val)
    if int(train_batch_count_per_epoch) < train_batch_count_per_epoch: train_batch_count_per_epoch = int(train_batch_count_per_epoch)+1
    val_batch_count_per_epoch: float = float(VAL_LEN) / float(BATCH_SIZE) / len(accuracies_val)
    if int(val_batch_count_per_epoch) < val_batch_count_per_epoch: val_batch_count_per_epoch = int(val_batch_count_per_epoch)+1
    train_loss_avg_per_epoch = [sum(losses_train[i:i+train_batch_count_per_epoch])/len(losses_train[i:i+train_batch_count_per_epoch]) for i in range(0, len(losses_train), train_batch_count_per_epoch)]
    val_loss_avg_per_epoch = [sum(losses_val[i:i+val_batch_count_per_epoch])/len(losses_val[i:i+val_batch_count_per_epoch]) for i in range(0, len(losses_val), val_batch_count_per_epoch)]
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_avg_per_epoch, label="Training Loss")
    plt.plot(val_loss_avg_per_epoch, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss (per epoch), Test Accuracy {accuracy_test[-1]}")
    plt.grid()
    plt.show()

    # accuracy
    plt.figure(figsize=(10,5))
    plt.plot(accuracies_train, label="Training Accuracy")
    plt.plot(accuracies_val, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy, Test Accuracy {accuracy_test[-1]}")
    plt.grid()
    plt.show()

def run_model_on_dataset(model, dataloader, loss_criterion, losses, accuracies, device, name="test", label_enc=None, label_dec=None, olala=None):
    accuracy = 0
    correct = 0
    total = 0
    losses_e = []
    with torch.no_grad():
        model.eval()
        for inp, label in dataloader:
            inp = inp.to(device)
            label = label.to(device)
            if olala:
                inp = label

            original_label = None
            if label_enc:
                label = label_enc(label)
                original_label = copy.deepcopy(label)

            out = model(inp)
            loss = loss_criterion(out, label)
            losses_e.append(loss.item())

            if label_dec:
                out = label_dec(out)
            if label_enc:
                label = original_label

            preds = torch.zeros_like(out)
            preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
            pred_classes = preds.argmax(dim=1)
            true_classes = label.argmax(dim=1)
            correct += (pred_classes == true_classes).sum().item()

            total += label.size(0)
        accuracy = correct / total
        model.train()

    if losses is not None:
        losses.extend(losses_e)
    accuracies.append(accuracy)

    print(f"{name} avg loss: {np.mean(losses_e)} (accuracy: {accuracies[-1]})")
    return losses, accuracies

def train_model(model, train_dataloader, val_dataloader, device, loss_criterion, opt, train_indices, val_indices, losses_train, accuracies_train, losses_val, accuracies_val, label_enc=None, label_dec=None, olala=None):
    # train
    highest_acc_train = 0.0
    epochs = NR_EPOCHS
    for epoch in range(epochs):
        correct = 0
        total = 0

        losses_train_e = []
        model.train()
        train_loop = tqdm(train_dataloader, desc=f"epoch {epoch+1}/{epochs}")
        for inp, label in train_loop:
            inp = inp.to(device)
            label = label.to(device)
            if olala:
                inp = label

            original_label = None
            if label_enc:
                label = label_enc(label)
                original_label = copy.deepcopy(label)

            out = model(inp)

            loss = loss_criterion(out, label)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses_train_e.append(loss.item())

            if label_dec:
                out = label_dec(out)
            if label_enc:
                label = original_label

            preds = torch.zeros_like(out)
            preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
            pred_classes = preds.argmax(dim=1)
            true_classes = label.argmax(dim=1)
            correct += (pred_classes == true_classes).sum().item()

            total += label.size(0)

        accuracy = correct / total

        if accuracy > highest_acc_train:
            highest_acc_train = accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "BATCH_SIZE": BATCH_SIZE,
                "train_indices": train_indices,
                "val_indices": val_indices,
            }, OUTPUT_DIR + "out.pth")

        losses_train.extend(losses_train_e)
        accuracies_train.append(accuracy)
        print(f"training avg loss: {np.mean(losses_train_e)} (accuracy: {accuracies_train[-1]})")

        #  validation
        lval, aval = [], []
        lval, aval = run_model_on_dataset(model, val_dataloader, loss_criterion, lval, aval, device, "validation")
        losses_val.extend(lval)
        accuracies_val.extend(aval)

        # lr scheduler
        # lr_scheduler.step(np.mean(aval))


def main():
    print("------------------------------------------------------------")
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda device name: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")

    global BATCH_SIZE
    BATCH_SIZE = int(input("> enter batch size: "))
    NR_EPOCHS = 0
    NR_EPOCHS = int(input("> enter nr epochs: "))

    # dataset
    print("loading train/test datasets...")
    # train/val split
    train_indices = []
    val_indices = []
    dataset_for_split = VGDataset(DATASET_PROCESSED_PATH, "train")
    split_len = int(len(dataset_for_split)*(1-VALIDATION_PERCENTAGE))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_for_split, [split_len, len(dataset_for_split)-split_len])
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    # # add augmented samples
    train_dataset = VGDataset(DATASET_PROCESSED_PATH, "train", train_dataset.dataset, train_dataset.indices)
    test_dataset = VGDataset(DATASET_PROCESSED_PATH, "test")

    # definitions
    # model = Classifier3D()

    autoencoder_label = AutoEncoderLabel()
    autoencoder_label_loss_criterion = nn.BCELoss()
    loss_criterion = nn.CrossEntropyLoss()

    model = Classifier3DEncodedLabel()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "max")

    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
    accuracy_test = []

    print("loading train/test dataloaders...")
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)
    print(f"train, val, and test datasets loaded, lenghts: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    model.to(device)
    print(f"model moved to device: {device} - {torch.cuda.get_device_name(torch.cuda.device)}")
    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    train_model(autoencoder_label, train_dataloader, val_dataloader, device, autoencoder_label_loss_criterion, opt, train_indices, val_indices, losses_train, accuracies_train, losses_val, accuracies_val, olala=True)
    label_enc = autoencoder_label.encoder
    label_dec = autoencoder_label.decoder
    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
    train_model(model, train_dataloader, val_dataloader, device, loss_criterion, opt, train_indices, val_indices, losses_train, accuracies_train, losses_val, accuracies_val, label_enc, label_dec)

    accuracy_test = []
    _, accuracy_test = run_model_on_dataset(model, test_dataloader, loss_criterion, None, accuracy_test, device)

    # save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    dummy = input("dummy (press enter)")
    save_filename = input("> Save the file to (just the file name, without extension): ")
    torch.save({
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "losses_train": losses_train,
        "losses_val": losses_val,
        "accuracies_train": accuracies_train,
        "accuracies_val": accuracies_val,
        "accuracy_test": accuracy_test,
        "BATCH_SIZE": BATCH_SIZE,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }, OUTPUT_DIR + save_filename + ".pth")
    # print(f"saving losses_val len: {len(losses_val)} last one: {losses_val[-1]}")
    # print(f"saving accuracies_val len: {len(accuracies_val)} last one: {accuracies_val[-1]}")
    print("Model saved.")

    dummy = input("dummy (press enter)")
    yes = input("> Plot results? (y/n)")
    if yes == "y":
        plot(losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test)

    print("done.")


if __name__ == "__main__":
    main()
