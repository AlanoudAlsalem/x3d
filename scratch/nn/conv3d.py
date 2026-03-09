"""
Conv3d layer (Module with parameters). Replaces torch.nn.Conv3d.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union
from scratch.nn.module import Module
from scratch.ops.conv3d import conv3d_forward


def _triple(v: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    return v


class Conv3d(Module):
    """
    3D convolution layer. Layout: (B, C, T, H, W).

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: (kT, kH, kW) or int.
        stride: (st, sh, sw) or int; default 1.
        padding: (pt, ph, pw) or int; default 0.
        bias: If True, learn bias.
        groups: 1 = standard conv; in_channels = depthwise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.groups = groups
        kT, kH, kW = self.kernel_size
        c_per_group = in_channels // groups
        # weight: (out_channels, in_channels//groups, kT, kH, kW)
        self._parameters["weight"] = np.zeros(
            (out_channels, c_per_group, kT, kH, kW), dtype=np.float32
        )
        self._init_weight()
        if bias:
            self._parameters["bias"] = np.zeros(out_channels, dtype=np.float32)
        else:
            self._parameters["bias"] = None

    def _init_weight(self) -> None:
        """Xavier-like init for 3D conv."""
        w = self._parameters["weight"]
        fan_in = np.prod(w.shape[1:])
        bound = (1.0 / fan_in) ** 0.5
        w[:] = np.random.uniform(-bound, bound, size=w.shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return conv3d_forward(
            x,
            self._parameters["weight"],
            self._parameters["bias"],
            self.stride,
            self.padding,
            self.groups,
        )
