"""
X3D Head: ProjectedPool (pre_conv, pool, post_conv) + Dropout + Linear + output_pool.
"""

from __future__ import annotations
import numpy as np
from scratch.nn.module import Module
from scratch.nn.conv3d import Conv3d
from scratch.nn.batchnorm3d import BatchNorm3d
from scratch.ops.activations import relu
from scratch.ops.pooling import avg_pool3d_forward
from scratch.ops.linear import linear_forward
from scratch.ops.dropout import dropout_forward


class ProjectedPool(Module):
    """
    pre_conv (192->432) + BN + ReLU -> AvgPool3d(16,7,7) -> post_conv (432->2048) + ReLU.
    [B,192,16,7,7] -> [B,2048,1,1,1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.pre_conv = Conv3d(192, 432, kernel_size=1, bias=False)
        self.pre_norm = BatchNorm3d(432, eps=1e-5, momentum=0.1)
        self.post_conv = Conv3d(432, 2048, kernel_size=1, bias=False)
        self._modules["pre_conv"] = self.pre_conv
        self._modules["pre_norm"] = self.pre_norm
        self._modules["post_conv"] = self.post_conv

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.pre_conv.forward(x)
        x = self.pre_norm.forward(x)
        x = relu(x)
        x = avg_pool3d_forward(x, kernel_size=(16, 7, 7))
        x = self.post_conv.forward(x)
        return relu(x)


class Head(Module):
    """
    Classification head: pool -> dropout -> linear(2048->num_classes) -> output_pool -> flatten.

    Args:
        num_classes: Number of output classes (default 400).
    """

    def __init__(self, num_classes: int = 400) -> None:
        super().__init__()
        self.pool = ProjectedPool()
        self._modules["pool"] = self.pool
        self.num_classes = num_classes
        self.dropout_p = 0.5
        # Linear: 2048 -> num_classes
        self._parameters["proj_weight"] = np.zeros((num_classes, 2048), dtype=np.float32)
        self._parameters["proj_bias"] = np.zeros(num_classes, dtype=np.float32)
        self._init_proj()

    def _init_proj(self) -> None:
        bound = (1.0 / 2048) ** 0.5
        self._parameters["proj_weight"][:] = np.random.uniform(-bound, bound, (self.num_classes, 2048))
        self._parameters["proj_bias"][:] = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.pool.forward(x)
        x = dropout_forward(x, self.dropout_p, self.training)
        # [B, 2048, 1, 1, 1] -> (B, 2048) for linear
        x = x.reshape(x.shape[0], -1)
        x = linear_forward(x, self._parameters["proj_weight"], self._parameters["proj_bias"])
        return x
