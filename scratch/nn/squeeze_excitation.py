"""
Squeeze-and-Excitation block: global pool -> two 1x1x1 convs -> scale.
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.conv3d import Conv3d
from scratch.ops.activations import relu, sigmoid
from scratch.ops.pooling import adaptive_avg_pool3d_forward
def _round_width(width: float, multiplier: float, min_width: int = 8, divisor: int = 8) -> int:
    """Round width*multiplier to nearest multiple of divisor, at least min_width."""
    w = width * multiplier
    w_out = max(min_width, int(w + divisor / 2) // divisor * divisor)
    if w_out < 0.9 * w:
        w_out += divisor
    return int(w_out)


class SqueezeExcitation(Module):
    """
    SE block: squeeze (global avg pool) -> excite (1x1x1 conv -> ReLU -> 1x1x1 conv -> Sigmoid) -> scale.

    Args:
        channels: Input/output channel count.
        se_ratio: Bottleneck ratio for mid channels (default 0.0625).
    """

    def __init__(self, channels: int, se_ratio: float = 0.0625) -> None:
        super().__init__()
        mid = _round_width(channels, se_ratio)
        self.conv1 = Conv3d(channels, mid, kernel_size=1, bias=True)
        self.conv2 = Conv3d(mid, channels, kernel_size=1, bias=True)
        self._modules["conv1"] = self.conv1
        self._modules["conv2"] = self.conv2

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Squeeze: [B,C,T,H,W] -> [B,C,1,1,1]
        scale = adaptive_avg_pool3d_forward(x, 1)
        # Excite: two 1x1x1 convs with ReLU and Sigmoid
        scale = self.conv1.forward(scale)
        scale = relu(scale)
        scale = self.conv2.forward(scale)
        scale = sigmoid(scale)
        return x * scale
