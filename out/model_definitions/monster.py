import torch
import torch.nn as nn
import copy

from data import plot_voxel_grid

class Classifier3DEncodedLabel(nn.Module):
    def __init__(self):
        super(Classifier3DEncodedLabel, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=2), # 25
            nn.BatchNorm3d(8),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1), # 23
            nn.AvgPool3d(2), # 11
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1), # 9
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(in_features=23328, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=5),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # add artificial channel
        x = x.unsqueeze(1)
        return self.net(x)

class AutoEncoderLabel(nn.Module):
    def __init__(self):
        super(AutoEncoderLabel, self).__init__()

        # input 40
        self.encoder = nn.Sequential(
            nn.Linear(in_features=40, out_features=20),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(in_features=20, out_features=5),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=5, out_features=20),
            nn.LeakyReLU(),
            nn.Linear(in_features=20, out_features=40),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # add artificial channel
        x = x.unsqueeze(1)
        out = self.decoder(self.encoder(x))

        if not self.training:   # eval mode
            out = (out > 0.5).float()

        return out

class AutoEncoder3D(nn.Module):
    def __init__(self):
        super(AutoEncoder3D, self).__init__()

        # input 51x51x51
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=2), # 4 * 25
            nn.LeakyReLU(),
            nn.BatchNorm3d(4),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1), # 8 * 23
            nn.LeakyReLU(),
            nn.MaxPool3d(2), # 8 * 11
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2), # 16 * 5
            nn.BatchNorm3d(16),
            nn.LeakyReLU()
            # nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1), # 32 * 1
            # nn.BatchNorm3d(32),
            # nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, output_padding=0), # 11
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), # 22

            nn.ConvTranspose3d(8, 4, kernel_size=3, stride=1), # 24
            nn.BatchNorm3d(4),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(4, 1, kernel_size=5, stride=2), # 49
            nn.Sigmoid()
        )

    def forward(self, x):
        # add artificial channel
        x = x.unsqueeze(1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Classifier3DWithEncoder(nn.Module):
    def __init__(self):
        super(Classifier3DWithEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=2), # 4 * 25
            nn.LeakyReLU(),
            nn.BatchNorm3d(4),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1), # 8 * 23
            nn.LeakyReLU(),
            nn.MaxPool3d(2), # 8 * 11
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2), # 16 * 5
            nn.BatchNorm3d(16),
            nn.LeakyReLU()
            # nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1), # 32 * 1
            # nn.BatchNorm3d(32),
            # nn.LeakyReLU()
        )

        self.feature_extractor_3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=2), # 25
            nn.BatchNorm3d(8),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1), # 23
            nn.AvgPool3d(2), # 11
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1), # 9
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25328, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=40)
        )

    def set_encoder(self, encoder):
        self.encoder = encoder

    def forward(self, x):
        # add artificial channel
        x = x.unsqueeze(1)
        encoded = None
        with torch.no_grad():
            encoded = torch.flatten(self.encoder(x), start_dim=1)

        features = torch.flatten(self.feature_extractor_3d(x), start_dim=1)

        concat = torch.cat((features, encoded), dim=-1)

        return self.classifier(concat)

class Classifier3D(nn.Module):
    def __init__(self):
        super(Classifier3D, self).__init__()

        # 33
        # self.net = nn.Sequential(
        #     # nn.Dropout(0.2),
        #     nn.Conv3d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 16
        #     nn.BatchNorm3d(40),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 14
        #     nn.BatchNorm3d(80),
        #     nn.LeakyReLU(),
        #     nn.MaxPool3d(2), # 80 * 7
        #     nn.Conv3d(in_channels=80, out_channels=128, kernel_size=3, stride=1), # 128 * 5
        #     nn.BatchNorm3d(128),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=16000, out_features=128),
        #     nn.Dropout(),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=128, out_features=40)
        # )

# ############ 51 ###############
#         self.feature_extractor_3d = nn.Sequential(
#             # nn.Dropout(0.2),
#             nn.Conv3d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 25
#             nn.BatchNorm3d(40),
#             nn.LeakyReLU(),
#             nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 23
#             nn.BatchNorm3d(80),
#             nn.LeakyReLU(),
#             nn.MaxPool3d(2), # 80 * 11
#             nn.Conv3d(in_channels=80, out_channels=128, kernel_size=3, stride=1), # 128 * 9
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(),
#         )
#         self.feature_extractor_2d = nn.Sequential(
#             # nn.Dropout(0.2),
#             nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 25
#             nn.BatchNorm2d(40),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 23
#             nn.BatchNorm2d(80),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2), # 80 * 11
#             nn.Conv2d(in_channels=80, out_channels=128, kernel_size=3, stride=1), # 128 * 9
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=124416, out_features=128),
#             nn.Dropout(),
#             nn.LeakyReLU(),
#             nn.Linear(in_features=128, out_features=40)
#         )
# ############ 51 ###############

########### 33 ###############
        self.feature_extractor_3d = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Conv3d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 16
            nn.BatchNorm3d(40),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 14
            nn.BatchNorm3d(80),
            nn.LeakyReLU(),
            nn.MaxPool3d(2), # 80 * 7
            nn.Conv3d(in_channels=80, out_channels=128, kernel_size=3, stride=1), # 128 * 5
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
        )
        self.feature_extractor_2d = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 16
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 14
            nn.BatchNorm2d(80),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 80 * 7
            nn.Conv2d(in_channels=80, out_channels=128, kernel_size=3, stride=1), # 128 * 5
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25600, out_features=128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=40)
        )
########### 33 ###############

        # self.net = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=40, kernel_size=5), # 6 * 29
        #     nn.LeakyReLU(),
        #     nn.BatchNorm3d(40),
        #     nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3), # 8 * 27
        #     nn.LeakyReLU(),
        #     nn.BatchNorm3d(80),
        #     nn.Conv3d(in_channels=80, out_channels=128, kernel_size=3, stride=2), # 16 * 13
        #     nn.LeakyReLU(),
        #     nn.BatchNorm3d(128),
        #     nn.MaxPool3d(2), # 128 * 6
        #     nn.Flatten(),
        #     nn.Linear(in_features=27648, out_features=128),
        #     nn.Dropout(),
        #     nn.Linear(in_features=128, out_features=40)
        # )

        # # input 51x51x51
        # input 33x33x33
        # voxnet
        # self.feature_extractor_3d = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1), # 4 * 49
        #     nn.Dropout(),
        #     nn.BatchNorm3d(4),
        #     nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1), # 8 * 47
        #     nn.Dropout(),
        #     nn.MaxPool3d(2), # 8 * 23
        #     nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=1), # 32 * 21
        #     nn.Dropout(),
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU()
        # )
        # # self.feature_extractor_3d_inv = copy.deepcopy(self.feature_extractor_3d)

        # self.feature_extractor_2d = nn.Sequential(
        #     nn.Conv2d(in_channels=672, out_channels=128, kernel_size=3, stride=2), # 128 * 10
        #     nn.Dropout(),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1), # 64 * 8
        #     nn.Dropout(),
        #     nn.MaxPool2d(2), # 4
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1), # 32 * 2
        #     nn.Dropout(),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU()
        # )
        # self.feature_extractor_2d_inv = copy.deepcopy(self.feature_extractor_2d)

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=1024),
        #     nn.Dropout(),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=1024, out_features=256),
        #     nn.Dropout(),
        #     nn.Linear(in_features=256, out_features=40),
        # )

        # self.classifier_2d = nn.Sequential(
        #     nn.Linear(in_features=2400, out_features=1024),
        #     nn.Dropout(),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=1024, out_features=256),
        #     nn.Dropout(),
        #     nn.Linear(in_features=256, out_features=40),
        # )

        # self.classifier_3d = nn.Sequential(
        #     nn.Linear(in_features=4000, out_features=1024),
        #     nn.Dropout(),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=1024, out_features=512),
        #     nn.Dropout(),
        #     nn.Linear(in_features=512, out_features=40),
        # )

        # self.net = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 16
        #     nn.BatchNorm3d(40),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 14
        #     nn.BatchNorm3d(80),
        #     nn.LeakyReLU(),
        #     nn.MaxPool3d(2), # 80 * 7
        #     nn.Conv3d(in_channels=80, out_channels=128, kernel_size=3, stride=1), # 128 * 5
        #     nn.BatchNorm3d(128),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=16000, out_features=128),
        #     nn.Dropout(),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=128, out_features=40)
        # )

        # permutation
        # self.feature_extractor = nn.Sequential(
        #     # nn.Dropout(0.2),
        #     nn.Conv3d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 25
        #     nn.BatchNorm3d(40),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 22
        #     nn.BatchNorm3d(80),
        #     nn.LeakyReLU(),
        #     nn.MaxPool3d(2), # 80 * 11
        #     nn.Conv3d(in_channels=80, out_channels=32, kernel_size=3, stride=2), # 32 * 5
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU()
        # )

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=4000, out_features=128),
        #     nn.Dropout(),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_features=128, out_features=40)
        # )

        # class Permute(nn.Module):
        #     def __init__(self, dims):
        #         super().__init__()
        #         self.dims = dims
        #     def forward(self, x):
        #         return torch.permute(x, self.dims)

        # perms = [
        #     (0, 1, 2, 3, 4),
        #     (0, 1, 2, 4, 3),
        #     (0, 1, 3, 2, 4),
        #     (0, 1, 3, 4, 2),
        #     (0, 1, 4, 2, 3),
        #     (0, 1, 4, 3, 2)
        # ]
        # self.permutes = []
        # for i in range(len(perms)):
        #     self.permutes.append(Permute(perms[i]))

    def forward(self, x):
        import numpy as np

        # add artificial channel
        x = x.unsqueeze(1)
        # return self.net(x)



        # inv_x = torch.sum(torch.stack((torch.ones(x.shape).cuda(), torch.neg(x).cuda()), dim=0).cuda(), dim=0).cuda()

        # features_3d = self.feature_extractor_3d(x)
        # features_3d = features_3d.view(features_3d.size(0), features_3d.size(1) * features_3d.size(2), features_3d.size(3), features_3d.size(4)) # treat depth slices as extra channels
        # features_2d = self.feature_extractor_2d(features_3d)
        # out = self.classifier(torch.flatten(features_2d, start_dim=1))
        # return out

        side1, _ = torch.max(x, axis=-1)
        side2, _ = torch.max(x, axis=-2)
        side3, _ = torch.max(x, axis=-3)

        # side1_inv, _ = torch.max(inv_x, axis=-1)
        # side2_inv, _ = torch.max(inv_x, axis=-2)
        # side3_inv, _ = torch.max(inv_x, axis=-3)

        # side1_rotated_90 = torch.rot90(side1, 1, (2, 3))
        # side1_rotated_180 = torch.rot90(side1, 2, (2, 3))
        # side1_rotated_m90 = torch.rot90(side1, -1, (2, 3))
        # side2_rotated_90 = torch.rot90(side2, 1, (2, 3))
        # side2_rotated_180 = torch.rot90(side2, 2, (2, 3))
        # side2_rotated_m90 = torch.rot90(side2, -1, (2, 3))
        # side3_rotated_90 = torch.rot90(side3, 1, (2, 3))
        # side3_rotated_180 = torch.rot90(side3, 2, (2, 3))
        # side3_rotated_m90 = torch.rot90(side3, -1, (2, 3))

        flat_3d = torch.flatten(self.feature_extractor_3d(x), start_dim=1)
        flat_2d_1 = torch.flatten(self.feature_extractor_2d(side1), start_dim=1)
        flat_2d_2 = torch.flatten(self.feature_extractor_2d(side2), start_dim=1)
        flat_2d_3 = torch.flatten(self.feature_extractor_2d(side3), start_dim=1)

        # inv_flat_3d = torch.flatten(self.feature_extractor_3d_inv(inv_x), start_dim=1)
        # inv_flat_2d_1 = torch.flatten(self.feature_extractor_2d_inv(side1_inv), start_dim=1)
        # inv_flat_2d_2 = torch.flatten(self.feature_extractor_2d_inv(side2_inv), start_dim=1)
        # inv_flat_2d_3 = torch.flatten(self.feature_extractor_2d_inv(side3_inv), start_dim=1)

        # flat_2d_1_90 = torch.flatten(self.feature_extractor_2d(side1_rotated_90), start_dim=1)
        # flat_2d_2_90 = torch.flatten(self.feature_extractor_2d(side2_rotated_90), start_dim=1)
        # flat_2d_3_90 = torch.flatten(self.feature_extractor_2d(side3_rotated_90), start_dim=1)
        # flat_2d_1_180 = torch.flatten(self.feature_extractor_2d(side1_rotated_180), start_dim=1)
        # flat_2d_2_180 = torch.flatten(self.feature_extractor_2d(side2_rotated_180), start_dim=1)
        # flat_2d_3_180 = torch.flatten(self.feature_extractor_2d(side3_rotated_180), start_dim=1)
        # flat_2d_1_m90 = torch.flatten(self.feature_extractor_2d(side1_rotated_m90), start_dim=1)
        # flat_2d_2_m90 = torch.flatten(self.feature_extractor_2d(side2_rotated_m90), start_dim=1)
        # flat_2d_3_m90 = torch.flatten(self.feature_extractor_2d(side3_rotated_m90), start_dim=1)

        # concatenated_features = torch.cat((flat_3d, flat_2d_1, flat_2d_2, flat_2d_3, flat_2d_1_90, flat_2d_2_90, flat_2d_3_90, flat_2d_1_180, flat_2d_2_180, flat_2d_3_180, flat_2d_1_m90, flat_2d_2_m90, flat_2d_3_m90), dim=-1)
        concatenated_features = torch.cat((flat_3d, flat_2d_1, flat_2d_2, flat_2d_3), dim=-1)
        # concatenated_features_inv = torch.cat((inv_flat_3d, inv_flat_2d_1, inv_flat_2d_2, inv_flat_2d_3), dim=-1)
        # concatenated_features = torch.cat((concatenated_features_normal, concatenated_features_inv), dim=-1)
        # concatenated_features = torch.cat((flat_3d, inv_flat_3d), dim=-1)

        # classification_2d = self.classifier_2d(torch.cat((flat_2d_1, flat_2d_2, flat_2d_3), dim=-1))
        # classification_3d = self.classifier_3d(flat_3d)

        return self.classifier(concatenated_features)

        # return torch.sum(torch.stack((classification_2d, classification_3d), dim=0), dim=0)

        # # if self.training:
        # #     return self.net(x)

        # # permute
        # permuted_cubes = []
        # for p in self.permutes:
        #     permuted_cubes.append(p(x))

        # # extract features
        # features = []
        # for perm in permuted_cubes:
        #     features.append(self.feature_extractor(perm))

        # # average pooling the 3d features
        # pooled_features, _ = torch.stack(features).max(dim=0) # _ is the index of the max
        # return self.classifier(pooled_features)
































#########################################################MAIN##########################################################



























import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy

from data import VGDataset, plot_voxel_grid, find_small_scale_vgs, get_label_str, modelnet40_label_to_idx
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
    model = Classifier3D()
    model.load_state_dict(checkpoint["model_state_dict"])
    opt_state_dict = checkpoint["opt_state_dict"]

    # optional fields
    losses_train = checkpoint.get("losses_train", [])
    losses_val = checkpoint.get("losses_val", [])
    accuracies_train = checkpoint.get("accuracies_train", [])
    accuracies_val = checkpoint.get("accuracies_val", [])
    accuracy_test = checkpoint.get("accuracy_test", [])
    train_indices = checkpoint.get("train_indices", [])
    val_indices = checkpoint.get("val_indices", [])

    if "BATCH_SIZE" in checkpoint:
        global BATCH_SIZE
        BATCH_SIZE = checkpoint["BATCH_SIZE"]
        print(f"set batch size: {BATCH_SIZE} because loaded model used it apparently")

    # print(f"loading losses_val len: {len(losses_val)} last one: {losses_val[-1]}")
    # print(f"loading accuracies_val len: {len(accuracies_val)} last one: {accuracies_val[-1]}")

    return model, losses_train, losses_val, accuracies_train, accuracies_val, accuracy_test, load_filename, opt_state_dict, train_indices, val_indices

# TODO: i think python takes arrays as reference, so no need to return (may be causing extra copy operations idk).
def run_model_on_dataset(model, dataloader, loss_criterion, losses, accuracies, device, name="test", last_epoch=False):
    accuracy = 0
    correct = 0
    total = 0
    losses_e = []


    all_preds = []
    all_labels = []
    from torchmetrics import ConfusionMatrix
    confmat = ConfusionMatrix(task="multiclass", num_classes=40).cuda()

    with torch.no_grad():
        model.eval()
        for inp, label in dataloader:
            inp = inp.to(device)
# comment/uncomment this for autoencoder
            # label = copy.deepcopy(inp).unsqueeze(1) # FOR AUTOENCODER
            label = label.to(device)

            outs = []
            # rotations = [0, 1, 2, 3]
            # for rotation in rotations:
            #     inp = rotate_3d_discrete(inp, rotation)
            #     out = model(inp)
            #     outs.append(out)
            # outs = torch.stack(outs, dim=0)
            # out = outs.mean(dim=0)
            # out = outs.sum(dim=0)

            out = model(inp)

            loss = loss_criterion(out, label)
            losses_e.append(loss.item())

# comment/uncomment this for autoencoder
            preds = torch.zeros_like(out)
            preds[torch.arange(out.size(0)), out.argmax(dim=1)] = 1
            pred_classes = preds.argmax(dim=1)
            true_classes = label.argmax(dim=1)
            if name=="test" or last_epoch:
                all_preds.append(pred_classes)
                all_labels.append(true_classes)
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

    if name=="test" or last_epoch:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        cm = confmat(all_preds, all_labels)
        print(cm)
        import matplotlib.pyplot as plt
        import seaborn as sns
        idx_to_label = {v: k for k, v in modelnet40_label_to_idx.items()}
        label_names = [idx_to_label[i] for i in range(40)]
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt="d", cmap="Blues", xticklabels=label_names,  yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

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

def rotate_3d_discrete(x, rotation):
    import random
    out = []
    for b in range(x.size(0)):
        cube = x[b]
        axes = (0, 1)
        rotated = torch.rot90(cube, k=rotation, dims=axes)
        out.append(rotated)

    return torch.stack(out, dim=0)

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

    opt = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.001)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=3)

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
        highest_acc_val = 0.0
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
                # inp = random_rotate_3d_discrete(inp)

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


            losses_train.extend(losses_train_e)
            accuracies_train.append(accuracy)
            print(f"training avg loss: {np.mean(losses_train_e)} (accuracy: {accuracies_train[-1]})")

            #  validation
            lval, aval = [], []
            lval, aval = run_model_on_dataset(model, val_dataloader, loss_criterion, lval, aval, device, "validation", last_epoch=(epoch==NR_EPOCHS-1))
            losses_val.extend(lval)
            accuracies_val.extend(aval)

            if np.mean(aval) > highest_acc_val:
                highest_acc_val = np.mean(aval)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "BATCH_SIZE": BATCH_SIZE,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                }, OUTPUT_DIR + "out.pth")

            lr_scheduler.step(np.mean(aval))

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
