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
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv3d(in_channels=1, out_channels=40, kernel_size=3, stride=2), # 40 * 25
            nn.BatchNorm3d(40),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=40, out_channels=80, kernel_size=3, stride=1), # 80 * 22
            nn.BatchNorm3d(80),
            nn.LeakyReLU(),
            nn.MaxPool3d(2), # 80 * 11
            nn.Conv3d(in_channels=80, out_channels=128, kernel_size=3, stride=2), # 128 * 5
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=16000, out_features=128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=40)
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
        # add artificial channel
        x = x.unsqueeze(1)

        return self.net(x)

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
