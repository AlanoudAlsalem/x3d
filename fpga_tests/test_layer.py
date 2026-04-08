"""
Single-layer FPGA-offload bring-up test.

Usage:
    python -m fpga_tests.test_layer                         # defaults to conv_b
    python -m fpga_tests.test_layer --layer conv_a
    python -m fpga_tests.test_layer --layer conv_b --seed 123
    python -m fpga_tests.test_layer --layer conv_b --out-dir fpga_tests/runs

What it does, step by step (matches §7 of fpga_flow.md):

  1. Build a Conv3d from the layer config, Xavier-init its weights.
  2. Generate a reproducible float32 input tensor from a fixed seed.
  3. Run the conv in float32 once — this is "Path A" and also the
     reference we use to pick the output scale.
  4. Compute symmetric int8 quantization parameters:
       s_in  : per-tensor input scale   (scalar)
       s_w   : per-output-channel weight scale (vector)
       s_out : per-tensor output scale  (scalar, from float output)
       M     : per-channel float multiplier (s_in * s_w) / s_out
       M0,n  : fixed-point form of M for the "FPGA" path
  5. Quantize input and weights to int8.
  6. Run TWO int8 convs:
       a. "software"  : float requantize   (ground truth for int8 path)
       b. "fpga sim"  : fixed-point requantize (stand-in for the FPGA)
  7. Dequantize both back to float and compare against each other and the
     float reference. Save every tensor to a single .npz for replay.

Scalability: nothing in this script knows what "conv_b" is. It just reads a
LayerConfig. To test conv_t / conv_xy / conv_a / conv_c, change --layer.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Make the project root importable when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scratch.nn.conv3d import Conv3d                        # noqa: E402
from fpga_tests.layer_configs import LAYER_CONFIGS          # noqa: E402
from fpga_tests import quant                                # noqa: E402
from fpga_tests.kernels import sw_int8_conv3d, fpga_sim_int8_conv3d  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-layer int8 FPGA bring-up test.")
    p.add_argument("--layer", default="conv_b", choices=sorted(LAYER_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="fpga_tests/runs")
    p.add_argument("--tol-lsb", type=int, default=2,
                   help="Max allowed HW<->SW int8 disagreement in LSBs.")
    return p.parse_args()


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = (a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rms": float(np.sqrt(np.mean(d * d))),
    }


def run(layer_name: str, seed: int, out_dir: str, tol_lsb: int = 2) -> int:
    cfg = LAYER_CONFIGS[layer_name]
    print(f"[fpga_tests] layer = {cfg.name}  ({cfg.description})")
    print(f"[fpga_tests] input  shape {cfg.input_shape}")
    print(f"[fpga_tests] kernel {cfg.kernel_size}  stride {cfg.stride}  "
          f"pad {cfg.padding}  groups {cfg.groups}  "
          f"{cfg.in_channels}->{cfg.out_channels}")

    # 1. Build the float Conv3d, deterministic weight init.
    np.random.seed(seed)
    conv = Conv3d(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        kernel_size=cfg.kernel_size,
        stride=cfg.stride,
        padding=cfg.padding,
        groups=cfg.groups,
        bias=False,
    )
    W = conv._parameters["weight"]          # (O, I/g, kT, kH, kW)

    # 2. Reproducible input tensor.
    rng = np.random.default_rng(seed)
    x_f32 = rng.standard_normal(cfg.input_shape).astype(np.float32)

    # 3. Float reference (Path A).
    y_ref_f32 = conv.forward(x_f32)
    print(f"[fpga_tests] output shape {y_ref_f32.shape}")

    # 4. Quantization parameters.
    s_in = quant.compute_tensor_scale(x_f32)
    s_w = quant.compute_weight_scales(W)
    s_out = quant.compute_tensor_scale(y_ref_f32)
    M = quant.compute_M(s_in, s_w, s_out)
    M0, n = quant.quantize_multiplier_fixed_point(M)
    print(f"[fpga_tests] s_in={float(s_in):.6g}  s_out={float(s_out):.6g}  "
          f"s_w[min,max]=[{float(s_w.min()):.3g},{float(s_w.max()):.3g}]  "
          f"M[min,max]=[{float(M.min()):.3g},{float(M.max()):.3g}]")

    # 5. Quantize input and weights.
    x_q = quant.quantize_tensor(x_f32, float(s_in))
    W_q = quant.quantize_weights(W, s_w)

    # 6. Two int8 convs.
    y_sw_q = sw_int8_conv3d(x_q, W_q, M, cfg.stride, cfg.padding, cfg.groups)
    y_hw_q = fpga_sim_int8_conv3d(x_q, W_q, M0, n, cfg.stride, cfg.padding, cfg.groups)

    # 7. Dequantize and compare.
    y_sw_f32 = quant.dequantize_tensor(y_sw_q, float(s_out))
    y_hw_f32 = quant.dequantize_tensor(y_hw_q, float(s_out))

    hw_vs_sw_int = diff_stats(y_hw_q, y_sw_q)
    hw_vs_sw_f = diff_stats(y_hw_f32, y_sw_f32)
    sw_vs_ref = diff_stats(y_sw_f32, y_ref_f32)
    hw_vs_ref = diff_stats(y_hw_f32, y_ref_f32)

    mismatched = int(np.sum(y_hw_q != y_sw_q))
    total = int(y_sw_q.size)
    print()
    print(f"[fpga_tests] HW vs SW (int8):  max_abs={hw_vs_sw_int['max_abs']:.0f}  "
          f"mismatched elements {mismatched}/{total} "
          f"({100.0 * mismatched / total:.3f}%)")
    print(f"[fpga_tests] HW vs SW (float): max_abs={hw_vs_sw_f['max_abs']:.4g}  "
          f"rms={hw_vs_sw_f['rms']:.4g}")
    print(f"[fpga_tests] SW vs FLOAT ref:  max_abs={sw_vs_ref['max_abs']:.4g}  "
          f"rms={sw_vs_ref['rms']:.4g}")
    print(f"[fpga_tests] HW vs FLOAT ref:  max_abs={hw_vs_ref['max_abs']:.4g}  "
          f"rms={hw_vs_ref['rms']:.4g}")

    # Pass/fail. For bring-up we accept the HW vs SW int8 diff being <= 1 LSB
    # (fixed-point rounding occasionally differs by one ulp from float rounding).
    passed = hw_vs_sw_int["max_abs"] <= float(tol_lsb)
    print(f"[fpga_tests] {'PASS' if passed else 'FAIL'}: "
          f"HW<->SW int8 max abs diff = {hw_vs_sw_int['max_abs']:.0f} "
          f"(tolerance: {tol_lsb} LSB)")

    # Dump everything to one .npz so a future run can replay the exact inputs.
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg.name}_seed{seed}.npz")
    np.savez_compressed(
        out_path,
        x_f32=x_f32,
        W=W,
        s_in=np.float32(s_in),
        s_w=s_w,
        s_out=np.float32(s_out),
        M=M,
        M0=M0,
        n=n,
        x_q=x_q,
        W_q=W_q,
        y_ref_f32=y_ref_f32,
        y_sw_q=y_sw_q,
        y_hw_q=y_hw_q,
        y_sw_f32=y_sw_f32,
        y_hw_f32=y_hw_f32,
    )
    meta = {
        "layer": cfg.name,
        "seed": seed,
        "in_shape": list(cfg.input_shape),
        "out_shape": list(y_ref_f32.shape),
        "hw_vs_sw_int": hw_vs_sw_int,
        "hw_vs_sw_float": hw_vs_sw_f,
        "sw_vs_ref_float": sw_vs_ref,
        "hw_vs_ref_float": hw_vs_ref,
        "mismatched_int8_elements": mismatched,
        "total_elements": total,
        "passed": passed,
    }
    with open(os.path.join(out_dir, f"{cfg.name}_seed{seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fpga_tests] saved tensors -> {out_path}")
    print(f"[fpga_tests] saved summary  -> {out_path.replace('.npz', '.json')}")
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    return run(args.layer, args.seed, args.out_dir, args.tol_lsb)


if __name__ == "__main__":
    raise SystemExit(main())
