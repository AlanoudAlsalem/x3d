"""
X3D Stem: (2+1)D factorized conv (spatial 1x3x3 then temporal 5x1x1) + BN + ReLU.
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.conv3d import Conv3d
from scratch.nn.batchnorm3d import BatchNorm3d
from scratch.ops.activations import relu


class Conv2plus1dStem(Module):
    """
    (2+1)D stem: conv_t (1x3x3 spatial, stride (1,2,2)) then conv_xy (5x1x1 temporal, depthwise).
    Input [B,3,16,224,224] -> [B,24,16,112,112].
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv_t = Conv3d(
            3, 24,
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
            groups=1, bias=False,
        )
        self.conv_xy = Conv3d(
            24, 24,
            kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0),
            groups=24, bias=False,
        )
        self._modules["conv_t"] = self.conv_t
        self._modules["conv_xy"] = self.conv_xy

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv_t.forward(x)
        x = self.conv_xy.forward(x)
        return x


class Stem(Module):
    """
    Full stem: Conv2plus1dStem -> BatchNorm3d(24) -> ReLU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = Conv2plus1dStem()
        self.norm = BatchNorm3d(24, eps=1e-5, momentum=0.1)
        self._modules["conv"] = self.conv
        self._modules["norm"] = self.norm

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv.forward(x)
        x = self.norm.forward(x)
        return relu(x)
