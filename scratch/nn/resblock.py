"""
Residual block: BottleneckBlock + skip connection; output = ReLU(bottleneck(x) + shortcut(x)).
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.bottleneck import BottleneckBlock
from scratch.nn.conv3d import Conv3d
from scratch.nn.batchnorm3d import BatchNorm3d
from scratch.ops.activations import relu


class ResBlock(Module):
    """
    ResBlock: branch2 = BottleneckBlock(x), shortcut = x or 1x1 conv+BN; out = ReLU(branch2 + shortcut).

    Args:
        in_channels: Input channels.
        inner_channels: Bottleneck inner channels.
        out_channels: Output channels.
        stride: Spatial stride (1 or 2).
        use_se: Use SE in bottleneck.
        se_ratio: SE ratio.
    """

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
        se_ratio: float = 0.0625,
    ) -> None:
        super().__init__()
        self.branch2 = BottleneckBlock(
            in_channels, inner_channels, out_channels,
            stride=stride, use_se=use_se, se_ratio=se_ratio,
        )
        self._modules["branch2"] = self.branch2
        self.has_branch1 = (in_channels != out_channels) or (stride != 1)
        self.has_branch1_norm = in_channels != out_channels
        if self.has_branch1:
            self.branch1_conv = Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=(1, stride, stride), padding=0, bias=False,
            )
            self._modules["branch1_conv"] = self.branch1_conv
        if self.has_branch1_norm:
            self.branch1_norm = BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
            self._modules["branch1_norm"] = self.branch1_norm

    def forward(self, x: np.ndarray) -> np.ndarray:
        residual = self.branch2.forward(x)
        if self.has_branch1:
            shortcut = self.branch1_conv.forward(x)
            if self.has_branch1_norm:
                shortcut = self.branch1_norm.forward(shortcut)
        else:
            shortcut = x
        return relu(residual + shortcut)
