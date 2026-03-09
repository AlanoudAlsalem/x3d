"""
ResStage: sequence of ResBlocks (same out_channels; first block can downsample).
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.resblock import ResBlock
from scratch.nn.sequential import ModuleList


class ResStage(Module):
    """
    A stage of residual blocks. First block uses stride and in_channels; rest use out_channels, stride=1.
    SE on even-indexed blocks (0, 2, 4, ...).

    Args:
        depth: Number of ResBlocks.
        in_channels: Input channels (for first block).
        inner_channels: Bottleneck inner channels.
        out_channels: Output channels for all blocks.
        stride: Spatial stride for first block (e.g. 2).
        se_ratio: SE ratio.
    """

    def __init__(
        self,
        depth: int,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
        stride: int = 2,
        se_ratio: float = 0.0625,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                ResBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    inner_channels=inner_channels,
                    out_channels=out_channels,
                    stride=stride if i == 0 else 1,
                    use_se=(i % 2 == 0),
                    se_ratio=se_ratio,
                )
            )
        self.res_blocks = ModuleList(blocks)
        self._modules["res_blocks"] = self.res_blocks

    def forward(self, x: np.ndarray) -> np.ndarray:
        for block in self.res_blocks:
            x = block.forward(x)
        return x
