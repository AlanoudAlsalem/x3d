"""
Fully connected (linear) layer (no PyTorch).
"""

from __future__ import annotations
import numpy as np


def linear_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Linear layer: out = x @ weight.T + bias.

    Args:
        x: Input shape (..., in_features). Last dimension is in_features.
        weight: Shape (out_features, in_features).
        bias: Shape (out_features,).

    Returns:
        Output shape (..., out_features).
    """
    out = np.dot(x, weight.T) + bias
    return out
