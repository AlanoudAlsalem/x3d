"""
Quantization accuracy validation for X3D-M (float32 vs int8).

This script measures how much accuracy the int8 quantization loses compared
to the float32 baseline, at two levels of granularity:

    Level 1 — Per-layer comparison
        For every QuantizedConv3d in the int8 model, run the same float32
        input through both the float32 Conv3d and the QuantizedConv3d, then
        compare outputs. Reports max/mean absolute error and cosine
        similarity per layer.

    Level 2 — End-to-end logit comparison
        Run N complete forward passes (same random inputs, or real clips
        from --calib-dir) through both models, and compare the final 400-
        class logit vectors. Reports top-1 / top-5 agreement, cosine
        similarity, and KL divergence.

Both levels require PRETRAINED weights for both models:
    - Float32:  weights/x3d_m_kinetics400.npz    (from convert_pytorch_weights_to_numpy.py)
    - Int8:     weights/x3d_m_int8.npz            (from quantize_x3d_ptq.py)

Usage
-----
    # Quick end-to-end check (10 random inputs, skip per-layer)
    python scripts/validate_quantization.py \\
        --fp32-weights weights/x3d_m_kinetics400.npz \\
        --int8-weights weights/x3d_m_int8.npz \\
        --num-inputs 10

    # Full validation with per-layer breakdown
    python scripts/validate_quantization.py \\
        --fp32-weights weights/x3d_m_kinetics400.npz \\
        --int8-weights weights/x3d_m_int8.npz \\
        --per-layer \\
        --num-inputs 50

    # Use real calibration clips instead of random inputs
    python scripts/validate_quantization.py \\
        --fp32-weights weights/x3d_m_kinetics400.npz \\
        --int8-weights weights/x3d_m_int8.npz \\
        --calib-dir data/kinetics_calib \\
        --per-layer

Notes
-----
- The per-layer test uses a TINY input (1, C, 4, 16, 16) to keep the
  reference int8 kernel fast. This is sufficient for error measurement
  but not for accuracy percentages.
- The end-to-end test uses the full (1, 3, 16, 224, 224) shape. With the
  reference int8 kernel this is VERY SLOW (~minutes per input). Use
  --num-inputs 5 for a quick sanity check.
- Results are saved to a JSON file alongside a human-readable console
  report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root (parent of scripts/) is on sys.path so that
# ``import scratch`` works regardless of how this script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors. Returns 0.0 on degenerate inputs."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def kl_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """
    KL(softmax(a) || softmax(b)) in nats, averaged over the batch.

    Uses log-sum-exp for numerical stability. A KL of 0 means the
    distributions are identical; values under 0.1 are typical for good
    quantization; above 1.0 suggests a problem.
    """
    # Softmax in float64 for stability.
    a = logits_a.astype(np.float64)
    b = logits_b.astype(np.float64)
    # Per-sample log-softmax.
    log_p = a - np.log(np.sum(np.exp(a - a.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - a.max(axis=-1, keepdims=True)
    log_q = b - np.log(np.sum(np.exp(b - b.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - b.max(axis=-1, keepdims=True)
    p = np.exp(log_p)
    kl = np.sum(p * (log_p - log_q), axis=-1)
    return float(np.mean(kl))


# ---------------------------------------------------------------------------
# Level 1: per-layer comparison
# ---------------------------------------------------------------------------

def collect_conv_layers(model, prefix: str = "") -> List[Tuple[str, object]]:
    """
    Walk a scratch Module and return (path, module) for every Conv3d or
    QuantizedConv3d.
    """
    from scratch.nn.conv3d import Conv3d
    from scratch.quantized.layers import QuantizedConv3d

    results = []
    if isinstance(model, (Conv3d, QuantizedConv3d)):
        results.append((prefix, model))
    for name, child in model._modules.items():
        child_prefix = f"{prefix}.{name}" if prefix else name
        results.extend(collect_conv_layers(child, child_prefix))
    return results


def run_per_layer(
    model_f32,
    model_int8,
) -> List[Dict]:
    """
    Compare every conv layer in both models on a small random input sized
    to match each layer's expected in_channels (we use a tiny spatial size
    to keep the reference kernel fast).

    Returns a list of per-layer result dicts.
    """
    layers_f32 = collect_conv_layers(model_f32)
    layers_int8 = collect_conv_layers(model_int8)

    # Match by path. Both models have the same hierarchy.
    f32_by_path = {path: mod for path, mod in layers_f32}
    int8_by_path = {path: mod for path, mod in layers_int8}
    common = sorted(set(f32_by_path) & set(int8_by_path))

    results = []
    for i, path in enumerate(common):
        lf = f32_by_path[path]
        lq = int8_by_path[path]

        C_in = lf.in_channels
        # Use a tiny spatial size so the reference kernel finishes quickly.
        # T=4 is enough to exercise any temporal kernel up to kT=5 with
        # padding=2.
        x = np.random.randn(1, C_in, 4, 8, 8).astype(np.float32) * 0.5

        y_f32 = lf.forward(x)
        y_int8 = lq.forward(x)

        abs_err = np.abs(y_f32 - y_int8)
        cos = cosine_similarity(y_f32, y_int8)

        entry = {
            "path": path,
            "kernel": list(lf.kernel_size),
            "groups": lf.groups,
            "channels": f"{lf.in_channels}->{lf.out_channels}",
            "max_abs_error": float(abs_err.max()),
            "mean_abs_error": float(abs_err.mean()),
            "cosine_similarity": cos,
            "output_shape": list(y_f32.shape),
        }
        results.append(entry)

        status = "OK" if cos > 0.90 else "WARN" if cos > 0.70 else "BAD"
        print(
            f"  [{i+1:3d}/{len(common)}] {status} "
            f"cos={cos:.4f}  max_err={abs_err.max():.4f}  "
            f"mean_err={abs_err.mean():.4f}  {path}"
        )

    return results


# ---------------------------------------------------------------------------
# Level 2: end-to-end logit comparison
# ---------------------------------------------------------------------------

def load_inputs(
    calib_dir: Optional[Path],
    num_inputs: int,
) -> List[np.ndarray]:
    """
    Load or generate input tensors shaped (1, 3, 16, 224, 224).

    If calib_dir is provided, load .npy files from it (each expected to be
    (3, 16, 224, 224) float32). Otherwise generate random normal tensors
    seeded for reproducibility.
    """
    inputs = []
    if calib_dir is not None:
        files = sorted(calib_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {calib_dir}")
        for f in files[:num_inputs]:
            arr = np.load(f).astype(np.float32)
            assert arr.shape == (3, 16, 224, 224), f"Bad shape {arr.shape} in {f}"
            inputs.append(arr[np.newaxis])
    else:
        rng = np.random.RandomState(42)
        for _ in range(num_inputs):
            inputs.append(rng.randn(1, 3, 16, 224, 224).astype(np.float32))
    return inputs


def run_end_to_end(
    model_f32,
    model_int8,
    inputs: List[np.ndarray],
) -> Dict:
    """
    Run both models on the same inputs and compare logit outputs.

    Returns an aggregate result dict.
    """
    top1_agree = 0
    top5_overlaps = []
    cosines = []
    kl_divs = []
    max_abs_errors = []
    mean_abs_errors = []

    for i, x in enumerate(inputs):
        t0 = time.perf_counter()
        logits_f32 = model_f32.forward(x)
        t_f32 = time.perf_counter() - t0

        t0 = time.perf_counter()
        logits_int8 = model_int8.forward(x)
        t_int8 = time.perf_counter() - t0

        # Flatten to (num_outputs,) for top-k comparison. For a full model
        # forward this is already (400,); for stem-only it might be (C,T,H,W).
        flat_f32 = logits_f32[0].ravel()
        flat_int8 = logits_int8[0].ravel()

        # Top-1
        pred_f32 = int(np.argmax(flat_f32))
        pred_int8 = int(np.argmax(flat_int8))
        agree = pred_f32 == pred_int8
        if agree:
            top1_agree += 1

        # Top-5 overlap
        top5_f32 = set(np.argsort(-flat_f32)[:5].tolist())
        top5_int8 = set(np.argsort(-flat_int8)[:5].tolist())
        overlap = len(top5_f32 & top5_int8)
        top5_overlaps.append(overlap)

        # Cosine similarity of logit vectors
        cos = cosine_similarity(logits_f32, logits_int8)
        cosines.append(cos)

        # KL divergence
        kl = kl_divergence(logits_f32, logits_int8)
        kl_divs.append(kl)

        # Absolute error
        ae = np.abs(logits_f32 - logits_int8)
        max_abs_errors.append(float(ae.max()))
        mean_abs_errors.append(float(ae.mean()))

        tag = "AGREE" if agree else "DIFFER"
        print(
            f"  [{i+1:3d}/{len(inputs)}] {tag}  "
            f"top1: f32={pred_f32} int8={pred_int8}  "
            f"top5_overlap={overlap}/5  "
            f"cos={cos:.4f}  kl={kl:.4f}  "
            f"time: f32={t_f32:.1f}s int8={t_int8:.1f}s"
        )

    n = len(inputs)
    result = {
        "num_inputs": n,
        "top1_agreement": f"{top1_agree}/{n} ({100*top1_agree/n:.1f}%)",
        "top5_mean_overlap": f"{np.mean(top5_overlaps):.2f}/5",
        "cosine_similarity_mean": float(np.mean(cosines)),
        "cosine_similarity_min": float(np.min(cosines)),
        "kl_divergence_mean": float(np.mean(kl_divs)),
        "kl_divergence_max": float(np.max(kl_divs)),
        "max_abs_error_mean": float(np.mean(max_abs_errors)),
        "mean_abs_error_mean": float(np.mean(mean_abs_errors)),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare float32 and int8 X3D-M outputs for quantization accuracy."
    )
    parser.add_argument(
        "--fp32-weights", type=Path, required=True,
        help="Path to float32 .npz (from convert_pytorch_weights_to_numpy.py).",
    )
    parser.add_argument(
        "--int8-weights", type=Path, required=True,
        help="Path to int8 .npz (from quantize_x3d_ptq.py).",
    )
    parser.add_argument(
        "--num-inputs", type=int, default=10,
        help="Number of inputs for end-to-end comparison (default: 10).",
    )
    parser.add_argument(
        "--calib-dir", type=Path, default=None,
        help="Optional directory of .npy calibration clips (3,16,224,224).",
    )
    parser.add_argument(
        "--per-layer", action="store_true",
        help="Run Level 1 per-layer comparison (adds ~1-5 min for reference kernel).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save results to this JSON file.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=400,
    )
    parser.add_argument(
        "--stem-only", action="store_true",
        help="Only run through the Stem block (fast sanity check).",
    )
    args = parser.parse_args()

    # Lazy imports so --help is fast even without numpy.
    from scratch.models.x3d_m import X3D_M
    from scratch.load_weights import load_pretrained_numpy
    from scratch.quantized import build_quantized_x3d_m, load_int8_weights

    # ----- Build both models -----
    print("=" * 70)
    print("BUILDING FLOAT32 MODEL")
    print("=" * 70)
    model_f32 = X3D_M(num_classes=args.num_classes)
    load_pretrained_numpy(model_f32, args.fp32_weights, strict=False, verbose=False)
    model_f32.eval()
    print(f"  Loaded float32 weights from {args.fp32_weights}")

    print()
    print("=" * 70)
    print("BUILDING INT8 MODEL")
    print("=" * 70)
    model_int8 = build_quantized_x3d_m(
        num_classes=args.num_classes,
        weights_path=args.int8_weights,
        backend="reference",
        strict_weights=False,
        verbose=False,
    )
    model_int8.eval()
    print(f"  Loaded int8 weights from {args.int8_weights}")

    all_results = {}

    # ----- Level 1: Per-layer -----
    if args.per_layer:
        print()
        print("=" * 70)
        print("LEVEL 1: PER-LAYER COMPARISON")
        print("=" * 70)
        print("  (Using tiny inputs per layer; reference kernel on small tensors)")
        print()
        layer_results = run_per_layer(model_f32, model_int8)
        all_results["per_layer"] = layer_results

        # Summary
        cosines = [r["cosine_similarity"] for r in layer_results]
        bad = [r for r in layer_results if r["cosine_similarity"] < 0.90]
        print()
        print(f"  Layers tested: {len(layer_results)}")
        print(f"  Cosine similarity — min: {min(cosines):.4f}, "
              f"mean: {np.mean(cosines):.4f}, median: {np.median(cosines):.4f}")
        if bad:
            print(f"  WARNING: {len(bad)} layer(s) with cosine < 0.90:")
            for r in bad:
                print(f"    {r['path']}: cos={r['cosine_similarity']:.4f}")
        else:
            print("  All layers have cosine >= 0.90")

    # ----- Level 2: End-to-end -----
    print()
    print("=" * 70)
    print("LEVEL 2: END-TO-END LOGIT COMPARISON")
    print("=" * 70)

    if args.stem_only:
        print("  (--stem-only: comparing Stem outputs, not full model)")
        # Extract just block 0 (Stem) from each model.
        stem_f32 = model_f32._modules["blocks"]._modules["0"]
        stem_int8 = model_int8._modules["blocks"]._modules["0"]
        rng = np.random.RandomState(42)
        inputs = [rng.randn(1, 3, 16, 224, 224).astype(np.float32)
                  for _ in range(args.num_inputs)]
        e2e = run_end_to_end(stem_f32, stem_int8, inputs)
        # top-1/top-5 don't make sense for Stem output; relabel.
        e2e["note"] = "Stem-only: top-k metrics are on flattened channel argmax, not class labels."
    else:
        inputs = load_inputs(args.calib_dir, args.num_inputs)
        print(f"  Running {len(inputs)} inputs through FULL model. "
              f"This is slow with the reference int8 kernel.")
        print()
        e2e = run_end_to_end(model_f32, model_int8, inputs)

    all_results["end_to_end"] = e2e

    # ----- Summary -----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in e2e.items():
        print(f"  {k}: {v}")

    # ----- Interpretation guide -----
    print()
    print("-" * 70)
    print("HOW TO READ THESE NUMBERS:")
    print("-" * 70)
    print("""
  top1_agreement:       What fraction of inputs get the same predicted class.
                        > 90% is good for random inputs; > 95% for real clips.

  top5_mean_overlap:    Average overlap of top-5 predictions (out of 5).
                        > 4.0 is good.

  cosine_similarity:    How aligned the 400-dim logit vectors are.
                        > 0.95 is great, 0.90-0.95 is acceptable, < 0.90 is a
                        problem — check per-layer with --per-layer.

  kl_divergence:        How different the softmax distributions are (in nats).
                        < 0.1 is great, 0.1-0.5 is acceptable, > 1.0 is bad.

  max/mean_abs_error:   Raw logit differences. Scale depends on the model;
                        compare across runs, not in absolute terms.

  Per-layer cosine < 0.90 flags:
                        These are the layers where quantization error is
                        concentrated. Usually depthwise 3x3x3 convolutions
                        with few weights per channel. If many layers flag,
                        consider quantization-aware training (QAT).
""")

    # ----- Save -----
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
