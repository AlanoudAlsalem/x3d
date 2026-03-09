"""
Sequential and ModuleList (chain modules / list of modules).
"""

from __future__ import annotations
from typing import List, Union
import numpy as np
from scratch.nn.module import Module


class Sequential(Module):
    """
    Runs a sequence of modules in order. forward(x) = module_n(...(module_1(x))).
    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x: np.ndarray) -> np.ndarray:
        for m in self._modules.values():
            x = m.forward(x)
        return x


class ModuleList(Module):
    """
    Holds a list of modules (for iteration in forward, e.g. ResStage).
    """

    def __init__(self, modules: List[Module]) -> None:
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, i: int) -> Module:
        """Get module by index (e.g. model.blocks[0])."""
        return self._modules[str(i)]
