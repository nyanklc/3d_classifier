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
VOXEL_SIZE = 0.02
MODEL_GRID_SHAPE_FILE = DATASET_PROCESSED_PATH + "model_grid_shape.txt"

BATCH_SIZE = 64
NR_EPOCHS = 30

OUTPUT_DIR = "./out/"

def plot(losses_train, losses_test, accuracies_train, accuracies_test):
    # loss
    plt.figure(figsize=(10,5))
    plt.plot(losses_train, label="Training Loss")
    plt.plot(losses_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training/Testing Loss")
    plt.grid()
    plt.show()

    # accuracy
    plt.figure(figsize=(10,5))
    plt.plot(accuracies_train, label="Training Accuracy")
    plt.plot(accuracies_test, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training/Testing Accuracy")
    plt.grid()
    plt.show()

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
    accuracies_train = checkpoint["accuracies_train"]
    accuracies_test = checkpoint["accuracies_test"]

    return model, losses_train, losses_test, accuracies_train, accuracies_test


def test_model(model, test_dataloader, loss_criterion, losses_test, accuracies_test, device):
    losses_test_e = []
    accuracy = 0

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inp, label in test_dataloader:
            inp = inp.to(device)
            label = label.to(device)

            out = model(inp)

            loss = loss_criterion(out, label)
            losses_test_e.append(loss.item())

            preds = torch.zeros_like(out)
            preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
            pred_classes = preds.argmax(dim=1)
            true_classes = label.argmax(dim=1)
            correct += (pred_classes == true_classes).sum().item()
            total += label.size(0)
        accuracy = correct / total

    losses_test.append(np.mean(losses_test_e))
    accuracies_test.append(accuracy)
    print(f"test avg loss: {losses_test[-1]} (accuracy: {accuracies_test[-1]})")
    return losses_test, accuracies_test

def demo_model():
    import torch.nn.functional as F
    from data import modelnet40_label_to_idx
    from data import convert_mesh
    idx_to_label = {v: k for k, v in modelnet40_label_to_idx.items()}

    model, losses_train, losses_test, accuracies_train, accuracies_test = load_model()

    yes = input("> Plot results? (y/n) (default: n)")
    if yes == "": yes = "n"
    if yes == "y":
        plot(losses_train, losses_test, accuracies_train, accuracies_test)

    demo_input_file = input("> Enter path to mesh (or voxel grid npy) to predict: ").strip('"')
    inp = None
    if demo_input_file.endswith(".off"):
        model_grid_shape = np.loadtxt(MODEL_GRID_SHAPE_FILE)
        model_grid_shape = tuple(model_grid_shape.astype(int))
        inp = torch.from_numpy(convert_mesh(demo_input_file, VOXEL_SIZE, model_grid_shape)).float()
    elif demo_input_file.endswith(".npy"):
        inp = torch.from_numpy(np.load(demo_input_file, allow_pickle=True)).float()
    else:
        print(f"unknown input file {demo_input_file}")
        return

    # add batch + channel dimensions: (1, 1, 51, 51, 51)
    inp = inp.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1)[0] # convert to probabilities

    for idx, p in enumerate(probs):
        print(f"{idx_to_label[idx]}: {p.item()*100:.2f}%")

    pred_idx = probs.argmax().item()
    print(f"\nPrediction: {idx_to_label[pred_idx]} ({probs[pred_idx].item()*100:.2f}%)")


def main():
    print("DON'T PUT INVALID INPUTS, THERE ARE NO CHECKS")
    print("-----------------------------------------------")

    print("Modes:")
    print("1. Create a new model and train (tests during training).")
    print("2. Load an existing model and train (tests during training).")
    print("3. Create a new model and only train (tests during training).")
    print("4. Load an existing model and only train (tests during training).")
    print("5. Load an existing model and only test.")
    print("6. Demo existing model.")

    args_in = input("> Select (default: 6): ")
    if args_in == "": args_in = "6"

    args_load_model = False
    args_train_model = False
    args_test_model = False
    args_demo = False
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
        case "6":
            args_demo = True
        case _:
            exit()

    if args_demo:
        demo_model()
        exit()

    BATCH_SIZE = int(input("> enter batch size: "))
    NR_EPOCHS = 0
    if args_train_model:
        NR_EPOCHS = int(input("> enter nr epochs: "))


    print("------------------------------------------------------------")
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda device name: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")

    # dataset
    print("loading train/test datasets...")
    train_dataset = VGDataset(DATASET_PROCESSED_PATH, "train")
    test_dataset = VGDataset(DATASET_PROCESSED_PATH, "test")
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)
    print(f"train and test datasets loaded, lenghts: {len(train_dataset)}, {len(test_dataset)}")

    # definitions
    model = Classifier3D()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_criterion = nn.CrossEntropyLoss()
    losses_train = []
    losses_test = []
    accuracies_train = []
    accuracies_test = []

    if not args_load_model:
        print("a new model created")
    else:
        model, losses_train, losses_test, accuracies_train, accuracies_test = load_model()

    model.to(device)
    print(f"model moved to device: {device} - {torch.cuda.get_device_name(torch.cuda.device)}")

    # train (and test during training)
    if args_train_model:
        epochs = NR_EPOCHS
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            correct = 0
            total = 0
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

                preds = torch.zeros_like(out)
                preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
                pred_classes = preds.argmax(dim=1)
                true_classes = label.argmax(dim=1)
                correct += (pred_classes == true_classes).sum().item()
                total += label.size(0)
            accuracy = correct / total

            losses_train.append(np.mean(losses_train_e))
            accuracies_train.append(accuracy)
            print(f"training avg loss: {losses_train[-1]} (accuracy: {accuracies_train[-1]})")
            losses_test, accuracies_test = test_model(model, test_dataloader, loss_criterion, losses_test, accuracies_test, device)

    # test
    if args_test_model:
        nr_test = int(input("> nr of times to test: "))
        losses = []
        accuracies = []
        for _ in range(nr_test):
            losses = test_model(model, test_dataloader, loss_criterion, losses, accuracies, device)
        print(f"Average test loss: {np.mean(losses)}")
        print(f"Average test accuracy: {np.mean(accuracies)}")

    # save
    if args_train_model:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        save_filename = input("> Save the file to (just the file name, without extension): ")
        torch.save({
            "model_state_dict": model.state_dict(),
            "losses_train": losses_train,
            "losses_test": losses_test,
            "accuracies_train": accuracies_train,
            "accuracies_test": accuracies_test
        }, OUTPUT_DIR + save_filename + ".pth")
        print("Model saved.")

    yes = input("> Plot results? (y/n)")
    if yes == "y":
        plot(losses_train, losses_test, accuracies_train, accuracies_test)

    print("done.")


if __name__ == "__main__":
    main()
