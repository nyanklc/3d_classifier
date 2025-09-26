import torch
import torch.nn as nn

class Classifier3D(nn.Module):
    def __init__(self):
        super(Classifier3D, self).__init__()
        # input 51x51x51
        # self.net = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=6, kernel_size=5), # 6 * 47x47x47
        #     nn.BatchNorm3d(6),
        #     nn.Conv3d(in_channels=6, out_channels=8, kernel_size=3), # 8 * 45x45x45
        #     nn.BatchNorm3d(8),
        #     nn.Conv3d(in_channels=8, out_channels=3, kernel_size=3), # 3 * 43x43x43
        #     nn.BatchNorm3d(3),
        #     nn.MaxPool3d(4),
        #     nn.Flatten(),
        #     nn.LazyLinear(out_features=128),
        #     nn.Linear(in_features=128, out_features=40)
        # )

        # input 51x51x51
        # voxnet
        self.feature_extractor_3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=2), # 8 * 25
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1), # 16 * 23
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2), # 16 * 11
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 32 * 5
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.feature_extractor_2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2), # 8 * 25
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1), # 16 * 23
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 16 * 11
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 32 * 5
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=13600, out_features=128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=40)
        )

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

        side1, _ = torch.max(x, axis=-1)
        side2, _ = torch.max(x, axis=-2)
        side3, _ = torch.max(x, axis=-3)

        side1_rotated_90 = torch.rot90(side1, 1, (2, 3))
        side1_rotated_180 = torch.rot90(side1, 2, (2, 3))
        side1_rotated_m90 = torch.rot90(side1, -1, (2, 3))
        side2_rotated_90 = torch.rot90(side2, 1, (2, 3))
        side2_rotated_180 = torch.rot90(side2, 2, (2, 3))
        side2_rotated_m90 = torch.rot90(side2, -1, (2, 3))
        side3_rotated_90 = torch.rot90(side3, 1, (2, 3))
        side3_rotated_180 = torch.rot90(side3, 2, (2, 3))
        side3_rotated_m90 = torch.rot90(side3, -1, (2, 3))

        flat_3d = torch.flatten(self.feature_extractor_3d(x), start_dim=1)
        flat_2d_1 = torch.flatten(self.feature_extractor_2d(side1), start_dim=1)
        flat_2d_2 = torch.flatten(self.feature_extractor_2d(side2), start_dim=1)
        flat_2d_3 = torch.flatten(self.feature_extractor_2d(side3), start_dim=1)

        flat_2d_1_90 = torch.flatten(self.feature_extractor_2d(side1_rotated_90), start_dim=1)
        flat_2d_2_90 = torch.flatten(self.feature_extractor_2d(side2_rotated_90), start_dim=1)
        flat_2d_3_90 = torch.flatten(self.feature_extractor_2d(side3_rotated_90), start_dim=1)
        flat_2d_1_180 = torch.flatten(self.feature_extractor_2d(side1_rotated_180), start_dim=1)
        flat_2d_2_180 = torch.flatten(self.feature_extractor_2d(side2_rotated_180), start_dim=1)
        flat_2d_3_180 = torch.flatten(self.feature_extractor_2d(side3_rotated_180), start_dim=1)
        flat_2d_1_m90 = torch.flatten(self.feature_extractor_2d(side1_rotated_m90), start_dim=1)
        flat_2d_2_m90 = torch.flatten(self.feature_extractor_2d(side2_rotated_m90), start_dim=1)
        flat_2d_3_m90 = torch.flatten(self.feature_extractor_2d(side3_rotated_m90), start_dim=1)

        concatenated_features = torch.cat((flat_3d, flat_2d_1, flat_2d_2, flat_2d_3, flat_2d_1_90, flat_2d_2_90, flat_2d_3_90, flat_2d_1_180, flat_2d_2_180, flat_2d_3_180, flat_2d_1_m90, flat_2d_2_m90, flat_2d_3_m90), dim=-1)

        return self.classifier(concatenated_features)

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
