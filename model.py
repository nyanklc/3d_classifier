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
