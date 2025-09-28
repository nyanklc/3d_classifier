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
from model import Classifier3D, AutoEncoder3D, Classifier3DWithEncoder

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

def load_model():
    dummy = input("dummy (press enter)")
    load_filename = input("> Load model from (full or relative path to file) (defaults to ./out/out.pth): ")
    if load_filename == "":
        load_filename = "./out/out.pth"

    while not Path(load_filename).exists():
        print("file doesn't exist, input again")
        load_filename = input("> ")

    checkpoint = torch.load(load_filename, weights_only=False)
    model = Classifier3DWithEncoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    losses_train = checkpoint["losses_train"]
    losses_val = checkpoint["losses_val"]
    accuracies_train = checkpoint["accuracies_train"]
    accuracies_val = checkpoint["accuracies_val"]
    accuracy_test = checkpoint["accuracy_test"]
    opt_state_dict = checkpoint["opt_state_dict"]
    train_indices = checkpoint["train_indices"]
    val_indices = checkpoint["val_indices"]
    global BATCH_SIZE
    BATCH_SIZE = checkpoint["BATCH_SIZE"]
    print(f"set batch size: {BATCH_SIZE} because loaded model used it apparently")

    # print(f"loading losses_val len: {len(losses_val)} last one: {losses_val[-1]}")
    # print(f"loading accuracies_val len: {len(accuracies_val)} last one: {accuracies_val[-1]}")

    return model, losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test, load_filename, opt_state_dict, train_indices, val_indices

# TODO: i think python takes arrays as reference, so no need to return (may be causing extra copy operations idk).
def run_model_on_dataset(model, dataloader, loss_criterion, losses, accuracies, device, name="test"):
    accuracy = 0
    correct = 0
    total = 0
    losses_e = []
    with torch.no_grad():
        model.eval()
        for inp, label in dataloader:
            inp = inp.to(device)
# comment/uncomment this for autoencoder
            # label = copy.deepcopy(inp).unsqueeze(1) # FOR AUTOENCODER
            label = label.to(device)

            out = model(inp)
            loss = loss_criterion(out, label)
            losses_e.append(loss.item())

# comment/uncomment this for autoencoder
            preds = torch.zeros_like(out)
            preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
            pred_classes = preds.argmax(dim=1)
            true_classes = label.argmax(dim=1)
            correct += (pred_classes == true_classes).sum().item()
            # out_binarized = (out >= 0.5).float()
            # correct += (out_binarized == label).sum().item() / (51*51*51)

            total += label.size(0)
        accuracy = correct / total
        model.train()

    if losses is not None:
        losses.extend(losses_e)
    accuracies.append(accuracy)

    print(f"{name} avg loss: {np.mean(losses_e)} (accuracy: {accuracies[-1]})")
    return losses, accuracies

# on CPU
def demo_model():
    import torch.nn.functional as F
    from data import modelnet40_label_to_idx
    from data import convert_mesh
    idx_to_label = {v: k for k, v in modelnet40_label_to_idx.items()}

    model, losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test, _, _, _, _ = load_model()

    dummy = input("dummy (press enter)")
    yes = input("> Plot training results? (y/n) (default: n)")
    if yes == "": yes = "n"
    if yes == "y":
        plot(losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test)

    dummy = input("dummy (press enter)")
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

    # add batch dimension
    inp = inp.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1)[0] # convert to probabilities

    for idx, p in enumerate(probs):
        print(f"{idx_to_label[idx]}: {p.item()*100:.2f}%")

    pred_idx = probs.argmax().item()
    print(f"\nPrediction: {idx_to_label[pred_idx]} ({probs[pred_idx].item()*100:.2f}%)")

# on GPU
def benchmark():
    print("------------------------------------------------------------")
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda device name: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")

    import os
    import torch.nn.functional as F
    from data import modelnet40_label_to_idx
    from data import get_label_str, get_label_id, label_id_to_np
    idx_to_label = {v: k for k, v in modelnet40_label_to_idx.items()}

    model, losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test, model_filename, _, _, _ = load_model()

    dummy = input("dummy (press enter)")
    yes = input("> Plot training results? (y/n) (default: n)")
    if yes == "": yes = "n"
    if yes == "y":
        plot(losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test)

    dummy = input("dummy (press enter)")
    inpinp = input("> only benchmark on the test dataset? (y/n)")
    only_test = False
    if inpinp == "y":
        only_test = True

    import pandas as pd

    model.to(device)
    model.eval()
    rows = []  # store results for CSV
    correct, total = 0, 0

    with torch.no_grad():
        for root, _, files in os.walk(DATASET_PROCESSED_PATH):
            for file in files:
                if not file.endswith(".npy"):
                    continue

                parent_dir_name = os.path.basename(root)
                if only_test and parent_dir_name != "test": continue

                print(file)

                inp = torch.from_numpy(np.load(os.path.join(root, file), allow_pickle=True)).float()
                # add batch dimension
                inp = inp.unsqueeze(0)
                inp = inp.to(device)

                out = model(inp)
                label = torch.from_numpy(label_id_to_np(get_label_id(get_label_str(file))))
                label = label.to(device)

                preds = torch.zeros_like(out)
                preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
                pred_classes = preds.argmax(dim=1)
                true_classes = label.argmax(dim=0)

                correct += 1 if pred_classes == true_classes else 0
                total += 1

                probs = F.softmax(out, dim=1)[0]  # convert to probabilities

                # collect (label, prob) pairs
                entries = [
                    (idx_to_label[idx], p.item() * 100)
                    for idx, p in enumerate(probs)
                ]
                entries = sorted(entries, key=lambda x: x[1], reverse=True)
                entries = [(label, prob) for label, prob in entries if prob >= 0.005]
                entries = entries[:5]

                summary = " | ".join(f"{label}: {prob:.2f}%" for label, prob in entries)

                print(summary + f" -- actual: {get_label_str(file)}")

                rows.append({
                    "file": file,
                    "actual_label": get_label_str(file),
                    "predicted_label": entries[0][0] if entries else "N/A",
                    "prediction_confidence": entries[0][1] if entries else 0,
                    "topk_predictions": summary
                })

    accuracy = correct / total
    print(f"ACCURACY OVER THE WHOLE DATASET: {accuracy:.4f}")

    # dump to CSV
    df = pd.DataFrame(rows)

    # append accuracy as last row
    df.loc[len(df)] = {
        "file": "TOTAL",
        "actual_label": "",
        "predicted_label": "",
        "prediction_confidence": "",
        "topk_predictions": f"Accuracy: {accuracy:.4f}",
        "model_filename": model_filename
    }

    dummy = input("dummy (press enter)")
    dump_filename = input("> Enter filename for csv output (without extension): ")
    df.to_csv(OUTPUT_DIR + dump_filename + ".csv", index=False)
    print(f"Benchmark results saved to {OUTPUT_DIR + dump_filename + ".csv"}")


def get_group(fname):
    # nr_before_number = len(get_label_str(fname).split())
    # return "_".join(...)
    # TODO
    return None

def random_rotate_3d_discrete(x):
    import random
    out = []
    for b in range(x.size(0)):
        cube = x[b]
        axes = (0, 1)
        k = random.randint(0, 3)
        rotated = torch.rot90(cube, k=k, dims=axes)
        out.append(rotated)

    return torch.stack(out, dim=0)

def main():
    print("DON'T ENTER INVALID INPUTS, THERE ARE NO SANITY CHECKS")
    print("-----------------------------------------------")

    print("Modes:")
    print("1. Create a new model and train+test.") # TODO: is testing after train really necessary? already tests during training anyway lol
    print("2. Load an existing model and train+test.")
    print("3. Create a new model and train.")
    print("4. Load an existing model and train.")
    print("5. Load an existing model and test.")
    print("6. Demo existing model.")
    print("7. Test existing model over the whole dataset.")
    print("8. Find small scale voxel grids.")
    print("9. Plot voxel grid.")
    print("10. train autoencoder")

    args_in = input("> Select (default: 6): ")
    if args_in == "": args_in = "6"

    args_load_model = False
    args_train_model = False
    args_test_model = False
    args_autoencoder = False
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
            demo_model()
            exit()
        case "7":
            benchmark()
            exit()
        case "8":
            dummy = input("dummy (press enter)")
            global DATASET_PROCESSED_PATH
            train_dataset = VGDataset(DATASET_PROCESSED_PATH, "train")
            test_dataset = VGDataset(DATASET_PROCESSED_PATH, "test")

            kernel_size = 15

            train_small_scales = find_small_scale_vgs(train_dataset, kernel_size)
            print(f"train count: {len(train_small_scales)}")

            test_small_scales = find_small_scale_vgs(test_dataset, kernel_size)
            print(f"test count: {len(test_small_scales)}")

            exit()
        case "9":
            dummy = input("dummy (press enter)")
            filepath = input("> enter filepath (.npy): ")
            while filepath != "":
                vg = np.load(filepath)
                plot_voxel_grid(vg)
                filepath = input("> enter filepath (.npy): ")
            exit()
        case "10":
            args_autoencoder = True
            args_train_model = True
        case _:
            print("??")
            exit()

    dummy = input("dummy (press enter)")
    global BATCH_SIZE
    BATCH_SIZE = int(input("> enter batch size: "))
    NR_EPOCHS = 0
    if args_train_model:
        dummy = input("dummy (press enter)")
        NR_EPOCHS = int(input("> enter nr epochs: "))


    print("------------------------------------------------------------")
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda device name: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")

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

    # get autoencoder encoder
    autoencoder = None
    if not args_autoencoder:
        autoencoder = AutoEncoder3D()
        checkpoint_ae = torch.load("./out/AUTOENCODER.pth", weights_only=False)
        autoencoder.load_state_dict(checkpoint_ae["model_state_dict"])

    model = None
    if not args_autoencoder:
        model = Classifier3D()
# comment/uncomment this for autoencoder
        # model.set_encoder(autoencoder.encoder)
    else:
        model = AutoEncoder3D()

    opt = optim.NAdam(model.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)

    loss_criterion = None
    if args_autoencoder:
        loss_criterion = nn.BCELoss()
    else:
        loss_criterion = nn.CrossEntropyLoss()
    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
    accuracy_test = []

    if not args_load_model:
        print("a new model created")
    else:
        model, losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test, _, opt_state_dict, train_indices, val_indices = load_model()
        opt.load_state_dict(opt_state_dict)
        train_dataset = torch.utils.data.Subset(dataset_for_split, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_for_split, val_indices)
        train_dataset = VGDataset(DATASET_PROCESSED_PATH, "train", train_dataset.dataset, train_dataset.indices)

    print("loading train/test dataloaders...")
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)
    print(f"train, val, and test datasets loaded, lenghts: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    model.to(device)
    print(f"model moved to device: {device} - {torch.cuda.get_device_name(torch.cuda.device)}")
    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    # train
    if args_train_model:
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
                if args_autoencoder:
                    label = copy.deepcopy(inp).unsqueeze(1) # FOR AUTOENCODER
                label = label.to(device)

                # random rotation
                inp = random_rotate_3d_discrete(inp)

                out = model(inp)

                loss = loss_criterion(out, label)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                losses_train_e.append(loss.item())

                if not args_autoencoder:
                    preds = torch.zeros_like(out)
                    preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
                    pred_classes = preds.argmax(dim=1)
                    true_classes = label.argmax(dim=1)
                    correct += (pred_classes == true_classes).sum().item()
                else:
                    out_binarized = (out >= 0.5).float()
                    correct += (out_binarized == label).sum().item() / (51*51*51)

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

            lr_scheduler.step()

    accuracy_test = []
    _, accuracy_test = run_model_on_dataset(model, test_dataloader, loss_criterion, None, accuracy_test, device)

    # save
    if args_train_model:
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

    # test
    if args_test_model:
        _, accuracies = run_model_on_dataset(model, test_dataloader, loss_criterion, None, [], device)
        print(f"Average test accuracy: {np.mean(accuracies)}")

    dummy = input("dummy (press enter)")
    yes = input("> Plot results? (y/n)")
    if yes == "y":
        plot(losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test)

    print("done.")


if __name__ == "__main__":
    main()
