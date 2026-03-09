"""
X3D Bottleneck block: 1x1x1 (expand) -> 3x3x3 depthwise -> 1x1x1 (project) with BN and activations.
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.conv3d import Conv3d
from scratch.nn.batchnorm3d import BatchNorm3d
from scratch.nn.squeeze_excitation import SqueezeExcitation
from scratch.nn.sequential import Sequential
from scratch.ops.activations import relu, silu


class BottleneckBlock(Module):
    """
    Bottleneck: conv_a (1x1) -> BN -> ReLU -> conv_b (3x3 depthwise) -> BN -> [SE] -> SiLU -> conv_c (1x1) -> BN.

    Args:
        in_channels: Input channels.
        inner_channels: Channels after expand (and for depthwise).
        out_channels: Output channels after project.
        stride: Spatial stride for conv_b (1 or 2).
        use_se: Whether to use SqueezeExcitation after conv_b BN.
        se_ratio: SE bottleneck ratio.
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
        self.conv_a = Conv3d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.norm_a = BatchNorm3d(inner_channels, eps=1e-5, momentum=0.1)
        self.conv_b = Conv3d(
            inner_channels,
            inner_channels,
            kernel_size=3,
            stride=(1, stride, stride),
            padding=1,
            groups=inner_channels,
            bias=False,
        )
        if use_se:
            se_module = SqueezeExcitation(inner_channels, se_ratio)
        else:
            se_module = Identity()
        self.norm_b = Sequential(
            BatchNorm3d(inner_channels, eps=1e-5, momentum=0.1),
            se_module,
        )
        self.conv_c = Conv3d(inner_channels, out_channels, kernel_size=1, bias=False)
        self.norm_c = BatchNorm3d(out_channels, eps=1e-5, momentum=0.1)
        for name, m in [
            ("conv_a", self.conv_a),
            ("norm_a", self.norm_a),
            ("conv_b", self.conv_b),
            ("norm_b", self.norm_b),
            ("conv_c", self.conv_c),
            ("norm_c", self.norm_c),
        ]:
            self._modules[name] = m

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv_a.forward(x)
        x = self.norm_a.forward(x)
        x = relu(x)
        x = self.conv_b.forward(x)
        x = self.norm_b.forward(x)
        x = silu(x)
        x = self.conv_c.forward(x)
        x = self.norm_c.forward(x)
        return x


class Identity(Module):
    """No-op: returns input unchanged."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
