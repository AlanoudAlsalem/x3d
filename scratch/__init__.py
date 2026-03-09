"""
Scratch implementation of 3D CNN primitives (no PyTorch).

Provides Conv3d, BatchNorm3d, activations, pooling, and X3D-M building blocks
for deployment on systems where PyTorch is not available (e.g. RISC-V SoC).

Usage:
    from scratch import X3D_M
    model = X3D_M(num_classes=400)
    out = model.forward(x)  # x: numpy array [B, 3, 16, 224, 224]
"""

from scratch.nn.module import Module
from scratch.models.x3d_m import X3D_M
from scratch.load_weights import load_pretrained_numpy, load_pretrained_numpy_if_available

__all__ = ["Module", "X3D_M", "load_pretrained_numpy", "load_pretrained_numpy_if_available"]
