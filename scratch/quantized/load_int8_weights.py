"""
Loader for the int8 .npz files produced by scripts/quantize_x3d_ptq.py.

The .npz file is flat and uses keys of the form::

    <layer_path>.weight_q       (int8)
    <layer_path>.weight_scale   (float32, per-channel)
    <layer_path>.bias_q         (int32, optional)
    <layer_path>.input_scale    (float32 scalar)
    <layer_path>.output_scale   (float32 scalar)
    __meta__.*                  (metadata, ignored at load time)

The layer paths are PyTorch-model module paths (e.g. ``blocks.2.res_blocks.0.
branch2.conv_a``) and match the QuantizedX3D_M module hierarchy.

This loader does NOT reuse scratch.load_weights.load_pretrained_numpy because
that function assumes every parameter is float32. Mixed-dtype parameters
(int8 weights, int32 bias, float32 scales) would be corrupted by the
float32 cast in the existing loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from scratch.nn.module import Module
from scratch.quantized.layers import QuantizedConv3d, QuantizedLinear


# Suffixes we recognize in the int8 .npz as int8-layer parameters.
_QUANT_SUFFIXES = ("weight_q", "weight_scale", "bias_q", "input_scale", "output_scale")

# Suffixes we recognize as plain float32 parameters on their parent module
# (used for the un-quantized Head Linear's proj_weight / proj_bias, which
# live as numpy parameters on the scratch Head module itself).
_FP32_SUFFIXES = ("proj_weight", "proj_bias")

_EXPECTED_DTYPE = {
    "weight_q": np.int8,
    "weight_scale": np.float32,
    "bias_q": np.int32,
    "input_scale": np.float32,
    "output_scale": np.float32,
    "proj_weight": np.float32,
    "proj_bias": np.float32,
}


def _get_module_by_path(root: Module, path_parts: List[str]) -> Module:
    """Walk the scratch module tree by dotted path. Raises KeyError on miss."""
    current = root
    for part in path_parts:
        current = current._modules[part]
    return current


def _split_key(key: str) -> Tuple[str, str]:
    """
    Split 'blocks.1.res_blocks.0.branch2.conv_a.weight_q' into
    ('blocks.1.res_blocks.0.branch2.conv_a', 'weight_q').

    Returns ('', '') if the key doesn't end in one of the known suffixes.
    """
    for suffix in _QUANT_SUFFIXES + _FP32_SUFFIXES:
        tail = "." + suffix
        if key.endswith(tail):
            return key[: -len(tail)], suffix
    return "", ""


def load_int8_weights(
    model: Module,
    path: Union[str, Path],
    *,
    strict: bool = True,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load int8-quantized weights into a QuantizedX3D_M (or any model composed
    of QuantizedConv3d / QuantizedLinear modules).

    Parameters
    ----------
    model : Module
        A scratch model whose leaves are QuantizedConv3d / QuantizedLinear.
    path : str or Path
        Path to the .npz produced by scripts/quantize_x3d_ptq.py.
    strict : bool
        If True, raise if any quantized parameter in the model was not
        populated by the file.
    verbose : bool
        Print per-key progress.

    Returns
    -------
    missing, unexpected : lists of str
        ``missing`` is the set of ``<layer>.<suffix>`` entries the model
        expected but the file did not provide. ``unexpected`` is the set of
        file entries that did not match any layer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Int8 weights file not found: {path}")

    with np.load(path, allow_pickle=False) as f:
        keys = list(f.files)
        data: Dict[str, np.ndarray] = {k: np.array(f[k]) for k in keys}

    unexpected: List[str] = []
    loaded: set = set()

    for key, arr in data.items():
        if key.startswith("__meta__"):
            continue
        layer_path, suffix = _split_key(key)
        if not suffix:
            unexpected.append(key)
            continue
        path_parts = layer_path.split(".")
        try:
            mod = _get_module_by_path(model, path_parts)
        except KeyError:
            unexpected.append(key)
            continue
        # Float32 fallback suffixes (e.g. proj_weight / proj_bias on the
        # un-quantized Head Linear) are accepted on any module that has
        # them as a direct parameter. Quantized suffixes must land on a
        # QuantizedConv3d / QuantizedLinear.
        is_fp32_key = suffix in _FP32_SUFFIXES
        if not is_fp32_key and not isinstance(mod, (QuantizedConv3d, QuantizedLinear)):
            unexpected.append(key)
            continue
        if suffix not in mod._parameters:
            unexpected.append(key)
            continue

        # Enforce dtype. The PTQ exporter writes the correct dtypes already,
        # but we want to catch accidental float32-only loaders.
        expected_dtype = _EXPECTED_DTYPE[suffix]
        arr_cast = np.asarray(arr).astype(expected_dtype, copy=False)

        # Scalars (input_scale / output_scale) are stored as 0-d arrays.
        if suffix in ("input_scale", "output_scale"):
            mod._parameters[suffix] = np.float32(arr_cast.reshape(()).item())
        else:
            # Shape check against the module's current parameter buffer.
            target = mod._parameters[suffix]
            if target is None:
                # bias_q was None (layer built without bias); allow loading
                # a bias from file if present.
                mod._parameters[suffix] = arr_cast.copy()
            else:
                if target.shape != arr_cast.shape:
                    if verbose:
                        print(
                            f"  skip shape mismatch {key}: "
                            f"file {arr_cast.shape} vs model {target.shape}"
                        )
                    unexpected.append(key)
                    continue
                mod._parameters[suffix] = arr_cast.copy()

        loaded.add(key)
        if verbose:
            print(f"  loaded {key}")

    # Determine missing keys: every QuantizedConv3d/QuantizedLinear in the
    # model should have been fully populated.
    missing: List[str] = []

    def _walk(mod: Module, prefix: str) -> None:
        if isinstance(mod, (QuantizedConv3d, QuantizedLinear)):
            for suffix in _QUANT_SUFFIXES:
                if suffix not in mod._parameters:
                    continue
                if suffix == "bias_q" and mod._parameters["bias_q"] is None:
                    continue
                full = f"{prefix}.{suffix}" if prefix else suffix
                if full not in loaded:
                    missing.append(full)
        for name, child in mod._modules.items():
            _walk(child, f"{prefix}.{name}" if prefix else name)

    _walk(model, "")

    if strict and missing:
        raise KeyError(
            f"Missing int8 keys in weight file: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}"
        )

    return missing, unexpected
