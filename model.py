import torch.nn as nn

class Classifier3D(nn.Module):
    def __init__(self):
        super(Classifier3D, self).__init__()
        # input 51x51x51
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=6, kernel_size=(5, 5, 5), padding=(2, 2, 2)), # 6 * 51x51x51
            nn.BatchNorm3d(6),
            nn.Dropout(p=0.5),
            nn.Conv3d(in_channels=6, out_channels=8, kernel_size=(3, 3, 3)), # 8 * 49x49x49
            nn.BatchNorm3d(8),
            nn.MaxPool3d(4), # (49 - 4)/4 + 1 = 45/4 + 1 = 11.25 + 1 = 12.25 → floor → 12
            nn.Flatten(), # 8 * 12 * 12 * 12 = 13824
            nn.Linear(in_features=13824, out_features=128),
            nn.Linear(in_features=128, out_features=40)
        )

    def forward(self, x):
        x = x.unsqueeze(1) # add artificial channel

        # sigmoid breaks cross entropy loss calculations i think
        # s = nn.Sigmoid()
        # return s(self.net(x))

        return self.net(x)
