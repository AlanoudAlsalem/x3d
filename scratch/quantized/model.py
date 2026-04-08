"""
QuantizedX3D_M: the int8 (hybrid) version of the X3D-M model.

Strategy
--------
Rather than re-declaring the full model from scratch, this module builds a
normal float32 ``scratch.models.x3d_m.X3D_M`` and then rewrites its module
tree in place:

    1. Every ``scratch.nn.conv3d.Conv3d`` is replaced with a structurally
       identical ``scratch.quantized.layers.QuantizedConv3d``. The same
       constructor arguments (channels, kernel, stride, padding, groups,
       bias) are reused so the module hierarchy and paths are unchanged.
    2. Every ``scratch.nn.batchnorm3d.BatchNorm3d`` is replaced with an
       ``_IdentityModule``. This reflects the fact that BN has already been
       folded into the preceding Conv3d during PTQ. The BN modules still
       appear in the tree (so that their paths remain valid) but they do
       nothing at inference time.
    3. The final Linear projection in the Head (``proj_weight`` /
       ``proj_bias``) is NOT quantized. It stays float32. This matches the
       project's Phase-1 scope: only Conv3d is offloaded to the FPGA.

The result is a model whose external interface is identical to the float32
X3D_M — same input shape, same output shape, same module paths — but whose
convolution layers run quantize → int8 conv → dequantize internally.

Usage
-----
    from scratch.quantized import build_quantized_x3d_m, load_int8_weights

    model = build_quantized_x3d_m(num_classes=400, backend="reference")
    load_int8_weights(model, "weights/x3d_m_int8.npz")
    model.eval()
    logits = model.forward(x_f32)                   # x_f32: (1, 3, 16, 224, 224)

The float32 model (``scratch.models.x3d_m.X3D_M``) is untouched and can be
built and run independently in the same process.
"""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np

from scratch.nn.module import Module
from scratch.nn.conv3d import Conv3d
from scratch.nn.batchnorm3d import BatchNorm3d
from scratch.models.x3d_m import X3D_M
from scratch.quantized.layers import QuantizedConv3d


class _IdentityModule(Module):
    """
    No-op module that replaces folded BatchNorm3d.

    We keep an empty module (instead of just wiring the parent's forward
    around the BN) so the module-tree paths used by the int8 weight loader
    stay valid. This way, an .npz file that happens to include BN keys
    produced before the folding step will simply be ignored (those keys
    will show up as "unexpected" at load time rather than crashing).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


def _swap_conv_and_bn(parent: Module, backend: str) -> None:
    """
    Recursively rewrite ``parent``'s child modules:

        Conv3d       -> QuantizedConv3d (same shape / stride / padding)
        BatchNorm3d  -> _IdentityModule (BN already folded into preceding conv)

    The swap preserves the name each child is stored under both in
    ``parent._modules`` and as an instance attribute, so existing
    ``self.conv_a.forward(x)``-style calls in the parent's ``forward()``
    continue to work unchanged.
    """
    for name, child in list(parent._modules.items()):
        if isinstance(child, Conv3d):
            new = QuantizedConv3d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                bias=(child._parameters.get("bias") is not None),
                groups=child.groups,
                backend=backend,
            )
            parent._modules[name] = new
            # Many parent modules hold a direct attribute reference
            # (e.g. self.conv_a = Conv3d(...)) which is what their
            # forward() actually uses. Keep that in sync too.
            if hasattr(parent, name):
                setattr(parent, name, new)

        elif isinstance(child, BatchNorm3d):
            new_bn = _IdentityModule()
            parent._modules[name] = new_bn
            if hasattr(parent, name):
                setattr(parent, name, new_bn)

        else:
            # Recurse into everything else (ResStage, ResBlock, Bottleneck,
            # SqueezeExcitation, Stem, Head, ProjectedPool, ModuleList, ...).
            _swap_conv_and_bn(child, backend)


class QuantizedX3D_M(X3D_M):
    """
    Hybrid int8 X3D-M model.

    Inherits from :class:`scratch.models.x3d_m.X3D_M` purely so that the
    module hierarchy is built once by the base ``__init__`` and then
    rewritten in place. The class itself adds no new state; its only
    purpose is to give the quantized variant a distinct type for clarity in
    debugging and profiling.
    """

    def __init__(self, num_classes: int = 400, backend: str = "reference") -> None:
        super().__init__(num_classes=num_classes)
        _swap_conv_and_bn(self, backend=backend)


def build_quantized_x3d_m(
    num_classes: int = 400,
    weights_path: Optional[Union[str, Path]] = None,
    *,
    backend: str = "reference",
    strict_weights: bool = False,
    verbose: bool = False,
) -> QuantizedX3D_M:
    """
    Build the quantized X3D-M model and optionally load int8 weights.

    Parameters
    ----------
    num_classes : int
        Output classes (400 for Kinetics-400).
    weights_path : str or Path, optional
        Path to an int8 ``.npz`` file produced by
        ``scripts/quantize_x3d_ptq.py``. If ``None``, the model is returned
        with zero-initialized int8 weights (useful for structure tests).
    backend : str
        Backend for QuantizedConv3d: ``"reference"`` (NumPy software kernel,
        the default, used for validating the pipeline) or ``"fpga"`` (real
        hardware driver; not wired up yet).
    strict_weights : bool
        If True, raise on any missing quantized parameter. Default False
        because the Head Linear's float32 ``proj_weight`` / ``proj_bias``
        are not part of the int8 file and can be loaded separately.
    verbose : bool
        Print per-key loader progress.

    Returns
    -------
    QuantizedX3D_M
    """
    model = QuantizedX3D_M(num_classes=num_classes, backend=backend)
    if weights_path is not None:
        from scratch.quantized.load_int8_weights import load_int8_weights
        load_int8_weights(model, weights_path, strict=strict_weights, verbose=verbose)
    return model
