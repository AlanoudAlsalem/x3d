"""
3D Batch normalization (no PyTorch). Normalizes over (B, T, H, W) per channel.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def batchnorm3d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
    training: bool = False,
    momentum: float = 0.1,
) -> np.ndarray:
    """
    BatchNorm3d forward. Layout: (B, C, T, H, W).

    Args:
        x: Input (B, C, T, H, W).
        weight: Gamma, shape (C,).
        bias: Beta, shape (C,).
        running_mean: Running mean (C,). Updated if training.
        running_var: Running variance (C,). Updated if training.
        eps: Small constant for numerical stability.
        training: If True, use batch stats and update running stats.
        momentum: Momentum for running stats update.

    Returns:
        Normalized tensor same shape as x.
    """
    B, C, T, H, W = x.shape
    N = B * T * H * W

    if training:
        # Per-channel mean over (B,T,H,W)
        mean = np.mean(x, axis=(0, 2, 3, 4))  # (C,)
        var = np.var(x, axis=(0, 2, 3, 4)) + eps  # (C,)
        running_mean[:] = (1 - momentum) * running_mean + momentum * mean
        running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean
        var = running_var + eps

    # Normalize: (x - mean) / sqrt(var), then scale and shift
    x_norm = (x - mean.reshape(1, C, 1, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1, 1))
    out = weight.reshape(1, C, 1, 1, 1) * x_norm + bias.reshape(1, C, 1, 1, 1)
    return out
