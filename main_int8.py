"""
Int8 (hybrid) X3D-M inference entry point — Path B from fpga_flow.md.

This script is the int8 counterpart to ``main.py`` and is kept completely
separate so you can run either model independently:

    python main.py            # float32 path (Path A)
    python main_int8.py       # int8 hybrid path (Path B)

The int8 model shares no runtime state with the float32 model. It uses
``scratch.quantized.QuantizedX3D_M``, which is structurally identical to
``scratch.models.x3d_m.X3D_M`` but swaps every Conv3d for a QuantizedConv3d
and folds BatchNorm into an identity. The Head's final Linear is left in
float32 because only convolutions are offloaded to the FPGA in Phase 1.

Weights are loaded from an int8 ``.npz`` produced by
``scripts/quantize_x3d_ptq.py``, which performs BN folding, activation
calibration, and per-channel symmetric int8 quantization offline.

Example
-------

    python main_int8.py --weights weights/x3d_m_int8.npz
    python main_int8.py --weights weights/x3d_m_int8.npz --backend reference
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from scratch.quantized import build_quantized_x3d_m


def build_int8_x3d_m(
    num_classes: int = 400,
    weights_path: Optional[Union[str, Path]] = None,
    *,
    backend: str = "reference",
    verbose: bool = False,
):
    """
    Build the QuantizedX3D_M and optionally load int8 weights.

    Parameters
    ----------
    num_classes : int
        Number of classification outputs (400 for Kinetics-400).
    weights_path : str or Path, optional
        Path to an int8 ``.npz`` from ``scripts/quantize_x3d_ptq.py``.
    backend : str
        "reference" (NumPy software int8 kernel) or "fpga" (not yet wired).
    verbose : bool
        Print loader progress.
    """
    return build_quantized_x3d_m(
        num_classes=num_classes,
        weights_path=weights_path,
        backend=backend,
        strict_weights=False,
        verbose=verbose,
    )


def run_forward(model, x: np.ndarray) -> np.ndarray:
    """Single forward pass. ``x`` must be float32 (B, 3, 16, 224, 224)."""
    assert x.dtype == np.float32
    return model.forward(x)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Int8 X3D-M inference (hybrid float32/int8 FPGA path)."
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("weights/x3d_m_int8.npz"),
        help="Path to int8 .npz (default: weights/x3d_m_int8.npz)",
    )
    parser.add_argument(
        "--backend", choices=["reference", "fpga"], default="reference",
        help="Int8 conv backend (default: reference software kernel).",
    )
    parser.add_argument(
        "--num-classes", type=int, default=400,
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the model but do not run a forward pass. Useful for "
             "checking that weights load cleanly.",
    )
    args = parser.parse_args()

    print("Building QuantizedX3D_M ...")
    weights_path = args.weights if args.weights.is_file() else None
    if weights_path is None:
        print(f"  WARNING: weights file not found at {args.weights}. "
              f"Using zero-initialized int8 parameters; outputs will be meaningless.")

    model = build_int8_x3d_m(
        num_classes=args.num_classes,
        weights_path=weights_path,
        backend=args.backend,
        verbose=args.verbose,
    )
    model.eval()

    if args.dry_run:
        print("Dry run: model built successfully.")
        return 0

    # Random input to exercise the pipeline. Replace with real video frames
    # in production use.
    x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
    print("Running forward pass (this is SLOW — the reference kernel is "
          "pure NumPy nested loops and intended for correctness, not speed) ...")
    t0 = time.perf_counter()
    logits = run_forward(model, x)
    dt = time.perf_counter() - t0
    top5 = np.argsort(-logits[0])[:5]
    print(f"Done in {dt:.2f}s")
    print(f"Output shape: {logits.shape}")
    print(f"Top-5 predicted class indices: {top5.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
