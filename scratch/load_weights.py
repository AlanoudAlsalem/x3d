"""
Load pretrained weights from NumPy .npz files (no PyTorch). For use on SoC.

Weights are expected to have been produced by scripts/convert_pytorch_weights_to_numpy.py
on a machine with PyTorch. Keys in the .npz match the scratch module hierarchy, e.g.:
  blocks.0.conv.conv_t.weight, blocks.5.proj_weight
"""

from __future__ import annotations
import numpy as np
from typing import Union, Dict, Any
from pathlib import Path

from scratch.nn.module import Module


def _get_module_by_path(root: Module, path_parts: list[str]) -> Module:
    """
    Traverse root by path_parts (e.g. ["blocks", "0", "conv", "conv_t"]) to get the leaf module.
    """
    current = root
    for part in path_parts:
        current = current._modules[part]
    return current


def load_pretrained_numpy(
    model: Module,
    path_or_archive: Union[str, Path, Dict[str, np.ndarray]],
    *,
    strict: bool = True,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Load pretrained weights from a .npz file or a dict of name -> array (NumPy only).

    Use this on the SoC after building the model; the .npz file must have been
    generated on a laptop by scripts/convert_pytorch_weights_to_numpy.py.

    Args:
        model: Scratch model (e.g. X3D_M from build_x3d_m()).
        path_or_archive: Path to a .npz file, or a dict mapping key -> np.ndarray.
        strict: If True, raise KeyError when the model has a parameter not in the file.
        verbose: If True, print each loaded key.

    Returns:
        (missing_keys, unexpected_keys). missing_keys: model params not in file;
        unexpected_keys: file keys that did not match any model parameter.
    """
    if isinstance(path_or_archive, (str, Path)):
        with np.load(path_or_archive, allow_pickle=False) as f:
            data = {k: np.asarray(f[k], dtype=np.float32).copy() for k in f.files}
    else:
        data = {k: np.asarray(v, dtype=np.float32).copy() for k, v in path_or_archive.items()}

    missing: list[str] = []
    unexpected: list[str] = []

    # Build set of (path, param_name) that the model has
    def collect_params(module: Module, prefix: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for name, param in module._parameters.items():
            if param is not None:
                out.append((prefix, name))
        for name, sub in module._modules.items():
            out.extend(collect_params(sub, f"{prefix}.{name}" if prefix else name))
        return out

    model_params = collect_params(model, "")
    # Normalize: ("blocks.0.conv.conv_t", "weight") -> "blocks.0.conv.conv_t.weight"
    model_keys = {f"{p}.{n}" if p else n for p, n in model_params}

    for key, arr in data.items():
        arr = np.asarray(arr, dtype=np.float32)
        parts = key.split(".")
        if len(parts) < 2:
            unexpected.append(key)
            continue
        param_name = parts[-1]
        path_parts = parts[:-1]
        try:
            mod = _get_module_by_path(model, path_parts)
        except KeyError:
            unexpected.append(key)
            continue
        if param_name not in mod._parameters:
            unexpected.append(key)
            continue
        target = mod._parameters[param_name]
        if target.shape != arr.shape:
            if verbose:
                print(f"  skip shape mismatch {key}: file {arr.shape} vs model {target.shape}")
            unexpected.append(key)
            continue
        mod._parameters[param_name] = arr.copy()
        if verbose:
            print(f"  loaded {key} {arr.shape}")
        model_keys.discard(key)

    missing = sorted(model_keys)
    if strict and missing:
        raise KeyError(f"Missing keys in weight file: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return (missing, unexpected)


def load_pretrained_numpy_if_available(
    model: Module,
    path: Union[str, Path],
    *,
    verbose: bool = True,
) -> bool:
    """
    Load weights from path if the file exists. No-op if file not found (NumPy only).

    Convenient for SoC scripts: pass a path that may or may not exist (e.g. next to
    the script or in a mounted drive). Returns True if loaded, False otherwise.
    """
    path = Path(path)
    if not path.is_file():
        if verbose:
            print(f"Weights file not found: {path} (using random init)")
        return False
    missing, unexpected = load_pretrained_numpy(model, path, strict=False, verbose=verbose)
    if verbose:
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    return True
