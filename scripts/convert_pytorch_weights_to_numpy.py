"""
Convert PyTorch X3D-M state_dict to NumPy .npz for loading on SoC (no PyTorch there).

Run this on a machine that has PyTorch (e.g. laptop). Produces a single .npz file
that can be copied to the PolarFire RISC-V SoC and loaded with
scratch.load_weights.load_pretrained_numpy(model, path).

Usage:
  # From PyTorchVideo hub (requires network)
  python scripts/convert_pytorch_weights_to_numpy.py -o weights/x3d_m_kinetics400.npz

  # From a local .pth / .pt checkpoint (state_dict only)
  python scripts/convert_pytorch_weights_to_numpy.py -i path/to/x3d_m.pth -o weights/x3d_m.npz
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

# PyTorch only when running this script (not on SoC)
try:
    import torch
except ImportError:
    torch = None


def _pytorch_state_dict_to_numpy(state_dict: dict) -> dict[str, np.ndarray]:
    """Convert PyTorch state_dict to dict of numpy arrays (CPU, float32)."""
    out = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        out[k] = v.detach().cpu().float().numpy()
    return out


def _rewrite_key_for_scratch(key: str) -> str:
    """
    Rewrite PyTorch/PyTorchVideo state_dict keys to match scratch module hierarchy.

    - blocks.5.proj.weight -> blocks.5.proj_weight (Head stores proj in _parameters)
    - blocks.5.proj.bias -> blocks.5.proj_bias
    - *.norm_b.1.block.0.* -> *.norm_b.1.conv1.* (SE first conv)
    - *.norm_b.1.block.2.* -> *.norm_b.1.conv2.* (SE second conv)
    """
    if key == "blocks.5.proj.weight":
        return "blocks.5.proj_weight"
    if key == "blocks.5.proj.bias":
        return "blocks.5.proj_bias"
    if ".norm_b.1.block.0." in key:
        return key.replace(".norm_b.1.block.0.", ".norm_b.1.conv1.")
    if ".norm_b.1.block.2." in key:
        return key.replace(".norm_b.1.block.2.", ".norm_b.1.conv2.")
    return key


def convert_state_dict_to_scratch_format(state_dict: dict) -> dict[str, np.ndarray]:
    """
    Convert PyTorch state_dict to numpy dict with keys matching scratch modules.

    Args:
        state_dict: From model.state_dict() or torch.load(...).

    Returns:
        Dict mapping scratch-style keys to numpy arrays (float32).
    """
    numpy_dict = _pytorch_state_dict_to_numpy(state_dict)
    out = {}
    for k, v in numpy_dict.items():
        new_key = _rewrite_key_for_scratch(k)
        out[new_key] = v.astype(np.float32)
    return out


def load_pretrained_from_hub() -> dict:
    """Load X3D-M pretrained weights from PyTorchVideo hub. Requires network."""
    if torch is None:
        raise RuntimeError("PyTorch is required to load from hub. Install torch.")
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
    return model.state_dict()


def load_state_dict_from_file(path: str | Path) -> dict:
    """Load state_dict from a .pth / .pt file (e.g. torch.save(model.state_dict(), path))."""
    if torch is None:
        raise RuntimeError("PyTorch is required to load .pth files. Install torch.")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Expected state_dict or dict with 'state_dict'; got {type(ckpt)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch X3D-M weights to NumPy .npz for SoC."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("weights/x3d_m_kinetics400.npz"),
        help="Output .npz path (default: weights/x3d_m_kinetics400.npz)",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="Optional: input .pth/.pt checkpoint (state_dict). If not set, load from hub.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Do not print key count and path.",
    )
    args = parser.parse_args()

    if torch is None:
        print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        return 1

    if args.input is not None:
        if not args.input.is_file():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"Loading state_dict from {args.input} ...")
        state_dict = load_state_dict_from_file(args.input)
    else:
        if not args.quiet:
            print("Loading pretrained X3D-M from PyTorchVideo hub ...")
        state_dict = load_pretrained_from_hub()

    converted = convert_state_dict_to_scratch_format(state_dict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **converted)

    if not args.quiet:
        print(f"Saved {len(converted)} arrays to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
