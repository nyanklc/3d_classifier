import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data import VGDataset
from model import Classifier3D

# DATASET_PATH = "./data/ModelNet40/"
DATASET_PROCESSED_PATH = "./data/out/"
# VOXEL_SIZE = 0.02

BATCH_SIZE = 64
NR_EPOCHS = 30

OUTPUT_DIR = "./out/"

def load_model():
    load_filename = input("> Load model from (full or relative path to file) (defaults to ./out/out.pth): ")
    if load_filename == "":
        load_filename = "./out/out.pth"

    while not Path(load_filename).exists():
        print("file doesn't exist, input again")
        load_filename = input("> ")

    checkpoint = torch.load(load_filename, weights_only=False)
    model = Classifier3D()
    model.load_state_dict(checkpoint["model_state_dict"])
    losses_train = checkpoint["losses_train"]
    losses_test = checkpoint["losses_test"]

    return model, losses_train, losses_test


def test_model(model, test_dataloader, loss_criterion, losses_test, device):
    losses_test_e = []

    with torch.no_grad():
        model.eval()
        for inp, label in test_dataloader:
            inp = inp.to(device)
            label = label.to(device)

            out = model(inp)

            loss = loss_criterion(out, label)
            losses_test_e.append(loss.item())

    losses_test.append(np.mean(losses_test_e))
    print(f"test avg loss: {losses_test[-1]}")
    return losses_test




args_load_model = False
args_train_model = False
args_test_model = False
def main():
    print("DON'T PUT INVALID INPUTS, THERE ARE NO CHECKS")
    print("-----------------------------------------------")

    print("Modes:")
    print("1. Create a new model and train (tests during training).")
    print("2. Load an existing model and train (tests during training).")
    print("3. Create a new model and only train (tests during training).")
    print("4. Load an existing model and only train (tests during training).")
    print("5. Load an existing model and only test.")

    args_in = input("> Select: ")
    match args_in:
        case "1":
            args_train_model = True
            args_test_model = True
        case "2":
            args_load_model = True
            args_train_model = True
            args_test_model = True
        case "3":
            args_train_model = True
        case "4":
            args_load_model = True
            args_train_model = True
        case "5":
            args_load_model = True
            args_test_model = True
        case _:
            exit()

    BATCH_SIZE = int(input("> enter batch size: "))
    NR_EPOCHS = int(input("> enter nr epochs: "))


    print("------------------------------------------------------------")
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda device name: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")

    # dataset
    train_dataset = VGDataset(DATASET_PROCESSED_PATH, "train")
    test_dataset = VGDataset(DATASET_PROCESSED_PATH, "test")
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)
    print(f"train and test datasets loaded, lenghts: {len(train_dataset)}, {len(test_dataset)}")

    # definitions
    model = Classifier3D()
    opt = optim.Adam(model.parameters())
    loss_criterion = nn.CrossEntropyLoss()
    losses_train = []
    losses_test = []

    if not args_load_model:
        print("a new model created")
    else:
        model, losses_train, losses_test = load_model()

    model.to(device)
    print(f"model moved to device: {device} - {torch.cuda.get_device_name(torch.cuda.device)}")

    # train (and test during training)
    if args_train_model:
        epochs = NR_EPOCHS
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            losses_train_e = []
            model.train()
            for inp, label in train_dataloader:
                inp = inp.to(device)
                label = label.to(device)

                out = model(inp)

                loss = loss_criterion(out, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses_train_e.append(loss.item())

            losses_train.append(np.mean(losses_train_e))
            print(f"training avg loss: {losses_train[-1]}")
            losses_test = test_model(model, test_dataloader, loss_criterion, losses_test, device)

    # test
    if args_test_model:
        nr_test = int(input("> nr of times to test: "))
        losses = []
        for _ in range(nr_test):
            losses = test_model(model, test_dataloader, loss_criterion, losses, device)
        print(f"Average test loss: {np.mean(losses)}")

    # save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_filename = input("> Save the file to (just the file name, without extension): ")
    torch.save({
        "model_state_dict": model.state_dict(),
        "losses_train": losses_train,
        "losses_test": losses_test,
    }, OUTPUT_DIR + save_filename + ".pth")
    print("Model saved.")

    if args_train_model:
        # plot
        plt.figure(figsize=(10,5))
        if args_train_model: plt.plot(losses_train, label="Training Loss")
        if args_test_model: plt.plot(losses_test, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training/Testing Loss")
        plt.grid()
        plt.show()

    print("done.")


if __name__ == "__main__":
    main()
