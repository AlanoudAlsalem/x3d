"""
Dropout (no PyTorch). Zeros elements with probability p in training.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def dropout_forward(
    x: np.ndarray,
    p: float,
    training: bool,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Dropout: in training, zero each element with probability p and scale by 1/(1-p).

    Args:
        x: Input array.
        p: Probability of zeroing (e.g. 0.5).
        training: If False, return x unchanged.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        Same shape as x; masked and scaled if training, else x.
    """
    if not training or p == 0:
        return x
    if rng is None:
        rng = np.random.default_rng()
    mask = (rng.random(x.shape) >= p).astype(x.dtype)
    return x * mask / (1.0 - p)
