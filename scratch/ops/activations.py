"""
Activation functions (no PyTorch): ReLU, SiLU (Swish), Sigmoid.
"""

from __future__ import annotations
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x). Returns array of same shape."""
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: 1 / (1 + exp(-x)). Output in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish): x * sigmoid(x)."""
    return x * sigmoid(x)
