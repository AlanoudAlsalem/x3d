"""
BatchNorm3d layer (Module with parameters). Replaces torch.nn.BatchNorm3d.
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.ops.batchnorm3d import batchnorm3d_forward


class BatchNorm3d(Module):
    """
    3D Batch normalization. Normalizes over (B, T, H, W) per channel.

    Args:
        num_features: Channel count C.
        eps: Small constant for variance (default 1e-5).
        momentum: For running stats update (default 0.1).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self._parameters["weight"] = np.ones(num_features, dtype=np.float32)
        self._parameters["bias"] = np.zeros(num_features, dtype=np.float32)
        self._parameters["running_mean"] = np.zeros(num_features, dtype=np.float32)
        self._parameters["running_var"] = np.ones(num_features, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return batchnorm3d_forward(
            x,
            self._parameters["weight"],
            self._parameters["bias"],
            self._parameters["running_mean"],
            self._parameters["running_var"],
            eps=self.eps,
            training=self.training,
            momentum=self.momentum,
        )
