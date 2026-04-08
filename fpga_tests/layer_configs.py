"""
Conv layer specifications for the FPGA bring-up harness.

Every entry describes a single Conv3d exactly the way scratch.nn.Conv3d takes
it, plus the shape of the input tensor we will feed it. All five conv types
used in X3D-M are listed so the harness is trivially scalable: adding a new
conv type is one new dict entry.

Shapes follow the project convention (B, C, T, H, W). Biases are False
everywhere in X3D-M because every Conv3d is followed by BatchNorm3d, so we
leave bias=False here too. Once BN folding is in, the folded bias can be
plugged in as int32 inside the int8 kernel.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LayerConfig:
    name: str
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    padding: Tuple[int, int, int]
    groups: int
    input_shape: Tuple[int, int, int, int, int]   # (B, C, T, H, W)
    description: str


# Plausible shapes taken from Stage 2 of X3D-M (blocks[1].res_blocks[0]).
# Feel free to tweak — the harness does not care about realism, only shape
# consistency with the layer's (in_channels, stride, padding, groups).
LAYER_CONFIGS = {
    "conv_b": LayerConfig(
        name="conv_b",
        in_channels=54,
        out_channels=54,
        kernel_size=(3, 3, 3),
        stride=(1, 2, 2),
        padding=(1, 1, 1),
        groups=54,                       # depthwise
        input_shape=(1, 54, 16, 56, 56),
        description="Bottleneck 3x3x3 depthwise conv (Stage 2, first block).",
    ),
    "conv_a": LayerConfig(
        name="conv_a",
        in_channels=24,
        out_channels=54,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        groups=1,
        input_shape=(1, 24, 16, 56, 56),
        description="Bottleneck 1x1x1 expand conv (Stage 2).",
    ),
    "conv_c": LayerConfig(
        name="conv_c",
        in_channels=54,
        out_channels=24,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        groups=1,
        input_shape=(1, 54, 16, 28, 28),
        description="Bottleneck 1x1x1 project conv (Stage 2).",
    ),
    "conv_t": LayerConfig(
        name="conv_t",
        in_channels=3,
        out_channels=24,
        kernel_size=(1, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
        groups=1,
        input_shape=(1, 3, 16, 224, 224),
        description="Stem (2+1)D spatial conv (naming quirk: conv_t = 1x3x3).",
    ),
    "conv_xy": LayerConfig(
        name="conv_xy",
        in_channels=24,
        out_channels=24,
        kernel_size=(5, 1, 1),
        stride=(1, 1, 1),
        padding=(2, 0, 0),
        groups=24,                       # depthwise in time
        input_shape=(1, 24, 16, 112, 112),
        description="Stem (2+1)D temporal depthwise conv (naming quirk: conv_xy = 5x1x1).",
    ),
}
