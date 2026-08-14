import math
import torch
from torch import nn

class PrismGhost(nn.Module):
    """
    PrismGhost: Primary standard convolution + Cheap depthwise convolution.
    Features intrinsic feature generation paired with group-wise cheap feature generation.
    """
    def __init__(self, c1, c2, k=3, s=1, ratio=2):
        super().__init__()
        primary = math.ceil(c2 / ratio)
        cheap = primary * (ratio - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(c1, primary, k, s, k // 2, bias=False),
            nn.BatchNorm2d(primary),
            nn.SiLU(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(primary, cheap, 3, 1, 1, groups=primary, bias=False),
            nn.BatchNorm2d(cheap),
            nn.SiLU(),
        )
        self.c2 = c2

    def forward(self, x):
        y = self.primary(x)
        return torch.cat((y, self.cheap(y)), 1)[:, :self.c2]
