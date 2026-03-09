"""
Base module (replaces torch.nn.Module). Holds parameters and training flag.
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np


class Module:
    """
    Base class for all layers. Tracks parameters and training mode.

    Subclasses define _init_parameters() to register arrays (weights, biases,
    running mean/var) and implement forward(x) for the forward pass.
    """

    def __init__(self) -> None:
        self._parameters: Dict[str, np.ndarray] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training = True

    def _init_parameters(self) -> None:
        """Override to register parameters (e.g. self._parameters["weight"] = ...)."""
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Override to define forward pass. x and return are numpy arrays."""
        raise NotImplementedError

    def parameters(self) -> List[np.ndarray]:
        """Return list of all parameter arrays (including from submodules)."""
        out: List[np.ndarray] = []
        for v in self._parameters.values():
            out.append(v)
        for m in self._modules.values():
            out.extend(m.parameters())
        return out

    def train(self, mode: bool = True) -> "Module":
        """Set training mode (True) or eval (False). Affects BatchNorm and Dropout."""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode (training=False)."""
        return self.train(False)
