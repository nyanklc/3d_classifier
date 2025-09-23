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
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2), # 32 * 24
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1), # 32 * 22
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2), # 32 * 11
            nn.Flatten(),
            nn.Linear(in_features=42592, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=40)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # add artificial channel

        # sigmoid breaks cross entropy loss calculations i think
        # s = nn.Sigmoid()
        # return s(self.net(x))

        return self.net(x)
