import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier3D(nn.Module):
    def __init__(self):
        super(Classifier3D, self).__init__()