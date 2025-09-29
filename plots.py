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

FILTER = ""

SMA_WINDOW_SIZE = 20
def smooth_curve(values, window_size=SMA_WINDOW_SIZE):
    """Apply simple moving average smoothing."""
    if len(values) < window_size:
        return values  # don't smooth if too short
    return np.convolve(values, np.ones(window_size)/window_size, mode="valid")

def smooth_curve_ema(values, alpha=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return smoothed

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

# Define some linestyles to cycle through
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # last one is custom dash-dot
linewidths = [1.0, 1.2, 1.0, 1.2, 1.0]  # vary slightly for readability


for i in range(len(list_losses_train)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]

    smoothed = list_losses_train[i]
    batches = range(1, len(smoothed) + 1)
    plt.figure(figsize=(15, 8))
    plt.plot(
        batches,
        smoothed,
        label=f"{filenames[i][:-4]} (Test Accuracy {tacc})",
    )
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch.png")

    smoothed = list_losses_val[i]
    batches = range(1, len(smoothed) + 1)
    plt.figure(figsize=(15, 8))
    plt.plot(
        batches,
        smoothed,
        label=f"{filenames[i][:-4]} (Test Accuracy {tacc})",
    )
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Validation Loss")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch.png")

    epochs = range(1, len(list_accuracies_val[i]) + 1)
    plt.figure(figsize=(15, 8))
    plt.plot(
        epochs,
        list_accuracies_val[i],
        label=f"{filenames[i][:-4]} Validation Accuracy (Test Accuracy {tacc})",
    )
    plt.plot(
        range(1, len(list_accuracies_train[i]) + 1),
        list_accuracies_train[i],
        label=f"{filenames[i][:-4]} Train Accuracy (Test Accuracy {tacc})",
    )
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Validation/Train Accuracy")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy.png")

exit()



# --- Training Loss ---
plt.figure(figsize=(15, 8))
for i in range(len(list_losses_train)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
    smoothed = smooth_curve(list_losses_train[i])
    batches = range(1, len(smoothed) + 1)
    plt.plot(
        batches,
        smoothed,
        label=f"{filenames[i][:-4]} (Test Accuracy {tacc})",
        # linestyle=linestyles[i % len(linestyles)],
        linewidth=linewidths[i % len(linewidths)]
    )
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Training Loss (SMA with window size {SMA_WINDOW_SIZE})")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch_sma.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch_sma.png")

# no SMA
# --- Training Loss ---
plt.figure(figsize=(15, 8))
for i in range(len(list_losses_train)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
    smoothed = list_losses_train[i]
    batches = range(1, len(smoothed) + 1)
    plt.plot(
        batches,
        smoothed,
        label=f"{filenames[i][:-4]} (Test Accuracy {tacc})",
        # linestyle=linestyles[i % len(linestyles)],
        linewidth=linewidths[i % len(linewidths)]
    )
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Training Loss")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_training_loss_batch.png")

# --- Validation Loss ---
plt.figure(figsize=(15, 8))
for i in range(len(list_losses_val)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
    smoothed = smooth_curve(list_losses_val[i])
    batches = range(1, len(smoothed) + 1)
    plt.plot(
        batches,
        smoothed,
        label=f"{filenames[i][:-4]} (Test Accuracy {tacc})",
        # linestyle=linestyles[i % len(linestyles)],
        linewidth=linewidths[i % len(linewidths)]
    )
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Validation Loss (SMA with window size {SMA_WINDOW_SIZE})")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch_sma.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch_sma.png")

# no SMA
# --- Validation Loss ---
plt.figure(figsize=(15, 8))
for i in range(len(list_losses_val)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
    smoothed = list_losses_val[i]
    batches = range(1, len(smoothed) + 1)
    plt.plot(
        batches,
        smoothed,
        label=f"{filenames[i][:-4]} (Test Accuracy {tacc})",
        # linestyle=linestyles[i % len(linestyles)],
        linewidth=linewidths[i % len(linewidths)]
    )
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Validation Loss")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_validation_loss_batch.png")

# # --- Accuracy ---
# plt.figure(figsize=(15, 8))
# for i in range(len(list_accuracies_train)):
#     tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
#     if tacc == "":
#         tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
#     plt.plot(
#         list_accuracies_train[i],
#         label=f"Training Accuracy {filenames[i][:-4]} (Test Accuracy {tacc})",
#         linestyle=linestyles[i % len(linestyles)],
#         linewidth=linewidths[i % len(linewidths)]
#     )
#     plt.plot(
#         list_accuracies_val[i],
#         label=f"Validation Accuracy {filenames[i][:-4]} (Test Accuracy {tacc})",
#         # linestyle=linestyles[(i+1) % len(linestyles)],  # offset so train/val differ
#         linewidth=linewidths[i % len(linewidths)]
#     )
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Accuracy")
# plt.grid(True, which="both", linestyle="--", alpha=0.5)
# plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy.png", dpi=300, bbox_inches="tight")
# plt.close()
# print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy.png")

# --- Accuracy (Full: 0–50 Epochs) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Training Accuracy ---
for i in range(len(list_accuracies_train)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
    epochs = range(1, len(list_accuracies_train[i]) + 1)
    axes[0].plot(
        epochs,
        list_accuracies_train[i],
        label=f"{filenames[i][:-4]} (Test Acc {tacc})",
        linewidth=linewidths[i % len(linewidths)]
    )
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Training Accuracy (0–50 Epochs)")
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend()

# --- Validation Accuracy ---
for i in range(len(list_accuracies_val)):
    tacc = "" if len(list_accuracy_test[i]) == 0 else list_accuracy_test[i][-1]
    if tacc == "":
        tacc = "" if len(list_accuracies_test[i]) == 0 else list_accuracies_test[i][-1]
    epochs = range(1, len(list_accuracies_val[i]) + 1)
    axes[1].plot(
        epochs,
        list_accuracies_val[i],
        label=f"{filenames[i][:-4]} (Test Acc {tacc})",
        linewidth=linewidths[i % len(linewidths)]
    )
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy (0–50 Epochs)")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy_subplots_vertical.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy_subplots_vertical.png")


# --- Zoomed Accuracy (40–50 Epochs) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Training zoom
for i in range(len(list_accuracies_train)):
    epochs = range(1, len(list_accuracies_train[i]) + 1)
    axes[0].plot(
        epochs,
        list_accuracies_train[i],
        label=f"{filenames[i][:-4]}",
        linewidth=linewidths[i % len(linewidths)]
    )
axes[0].set_xlim(40, 50)
axes[0].set_ylim(0.80, 1.0)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Training Accuracy (Epochs 40–50)")
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend()

# Validation zoom
for i in range(len(list_accuracies_val)):
    epochs = range(1, len(list_accuracies_val[i]) + 1)
    axes[1].plot(
        epochs,
        list_accuracies_val[i],
        label=f"{filenames[i][:-4]}",
        linewidth=linewidths[i % len(linewidths)]
    )
axes[1].set_xlim(40, 50)
axes[1].set_ylim(0.80, 1.0)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy (Epochs 40–50)")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy_zoom_subplots_vertical.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("saved - " + OUTPUT_DIR + "plots/" + saveto + f"{filenames[i][:-4]}_accuracy_zoom_subplots_vertical.png")

