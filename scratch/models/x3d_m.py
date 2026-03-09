"""
X3D-M full model: Stem + 4 ResStages + Head. No PyTorch.
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.sequential import ModuleList
from scratch.nn.stem import Stem
from scratch.nn.resstage import ResStage
from scratch.nn.head import Head


class X3D_M(Module):
    """
    X3D-M: Stem -> Stage2 -> Stage3 -> Stage4 -> Stage5 -> Head.

    blocks[0] = Stem (Conv2plus1d + BN + ReLU)
    blocks[1] = Stage 2 (3 ResBlocks, 24ch, 112->56)
    blocks[2] = Stage 3 (5 ResBlocks, 48ch, 56->28)
    blocks[3] = Stage 4 (11 ResBlocks, 96ch, 28->14)
    blocks[4] = Stage 5 (7 ResBlocks, 192ch, 14->7)
    blocks[5] = Head (Conv5 + Pool + FC)

    Input: (B, 3, 16, 224, 224). Output: (B, num_classes).

    Args:
        num_classes: Number of classification outputs (default 400).
    """

    def __init__(self, num_classes: int = 400) -> None:
        super().__init__()
        self.blocks = ModuleList([
            Stem(),
            ResStage(depth=3, in_channels=24, inner_channels=54, out_channels=24, stride=2),
            ResStage(depth=5, in_channels=24, inner_channels=108, out_channels=48, stride=2),
            ResStage(depth=11, in_channels=48, inner_channels=216, out_channels=96, stride=2),
            ResStage(depth=7, in_channels=96, inner_channels=432, out_channels=192, stride=2),
            Head(num_classes=num_classes),
        ])
        self._modules["blocks"] = self.blocks

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (B, 3, 16, 224, 224), returns (B, num_classes)."""
        for block in self.blocks:
            x = block.forward(x)
        return x
