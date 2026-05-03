"""
Int8 (hybrid) X3D-M inference entry point — Path B from fpga_flow.md.

Uses the ``scratch.quantized`` package for a QuantizedX3D_M model that swaps
every Conv3d for a QuantizedConv3d and folds BatchNorm into an identity.

This script mirrors ``main.py`` in output format, profiling, and metrics so
that float32 and int8 runs can be compared on a common baseline.

    python main.py            # float32 path (Path A)
    python main_int8.py       # int8 hybrid path (Path B)

Run with profiling:
    python main_int8.py --profile --notes "Testing on MacBook Pro"
    python main_int8.py --profile --method fast
    python main_int8.py --profile --method threaded --notes "PolarFire SoC"
    python main_int8.py --profile --stem-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from scratch.quantized import build_quantized_x3d_m
from scratch.ops.conv3d_int8 import VALID_METHODS as INT8_METHODS
from scratch.stats import (
    StatsCollector,
    estimate_conv3d_flops,
    estimate_linear_flops,
    count_parameters,
)


def build_int8_x3d_m(
    num_classes: int = 400,
    weights_path: Optional[Union[str, Path]] = None,
    *,
    backend: str = "reference",
    method: Optional[str] = None,
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
    method : str, optional
        Int8 conv implementation from ``scratch.ops.conv3d_int8``:
        "slow", "fast", "threaded", or "native".  Overrides the reference
        kernel when set.
    verbose : bool
        Print loader progress.
    """
    return build_quantized_x3d_m(
        num_classes=num_classes,
        weights_path=weights_path,
        backend=backend,
        method=method,
        strict_weights=False,
        verbose=verbose,
    )


def run_forward(model, x: np.ndarray) -> np.ndarray:
    """Single forward pass. ``x`` must be float32 (B, 3, 16, 224, 224)."""
    assert x.dtype == np.float32
    return model.forward(x)


# ---------------------------------------------------------------------------
# Profiled forward — mirrors main.py run_forward_profiled exactly
# ---------------------------------------------------------------------------

def run_forward_profiled(
    model,
    x: np.ndarray,
    collector: StatsCollector,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run forward pass with detailed per-layer profiling.

    Walks the QuantizedX3D_M layer by layer (same structure as the float32
    model) so every QuantizedConv3d, Identity (fused BN), activation, pooling,
    and Linear gets its own timing entry.
    """
    from scratch.ops.activations import relu, silu, sigmoid
    from scratch.ops.pooling import avg_pool3d_forward, adaptive_avg_pool3d_forward
    from scratch.ops.linear import linear_forward
    from scratch.ops.dropout import dropout_forward

    if verbose:
        print("\n" + "=" * 60)
        print("PROFILED FORWARD PASS (Int8)")
        print("=" * 60)

    stem = model.blocks[0]
    stages = [model.blocks[i] for i in range(1, 5)]
    head = model.blocks[5]

    # ---------------------------------------------------------------- Stem
    collector.start_section("Stem")
    if verbose:
        print("\n[Stem] Processing...")

    input_shape = x.shape
    with collector.time_layer(
        "Stem.conv.conv_t", "QuantizedConv3d",
        input_shape,
        params=count_parameters(stem.conv.conv_t.weight_q, stem.conv.conv_t.bias_q),
        flops=estimate_conv3d_flops(input_shape, (1, 24, 16, 112, 112), (1, 3, 3), groups=1),
        kernel_size=(1, 3, 3), stride=(1, 2, 2), groups=1,
    ) as timer:
        x = stem.conv.conv_t.forward(x)
        timer.set_output_shape(x.shape)
    if verbose:
        print(f"  conv_t: {input_shape} -> {x.shape}")

    input_shape = x.shape
    with collector.time_layer(
        "Stem.conv.conv_xy", "QuantizedConv3d (depthwise)",
        input_shape,
        params=count_parameters(stem.conv.conv_xy.weight_q, stem.conv.conv_xy.bias_q),
        flops=estimate_conv3d_flops(input_shape, x.shape, (5, 1, 1), groups=24),
        kernel_size=(5, 1, 1), stride=1, groups=24,
    ) as timer:
        x = stem.conv.conv_xy.forward(x)
        timer.set_output_shape(x.shape)
    if verbose:
        print(f"  conv_xy: {input_shape} -> {x.shape}")

    input_shape = x.shape
    with collector.time_layer(
        "Stem.norm", "Identity (fused BN)",
        input_shape,
        params=0,
    ) as timer:
        x = stem.norm.forward(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer("Stem.relu", "ReLU", x.shape) as timer:
        x = relu(x)
        timer.set_output_shape(x.shape)

    collector.end_section("Stem")
    if verbose:
        print(f"  Stem output: {x.shape}")

    # ------------------------------------------------------------ Stages 2-5
    stage_configs = [
        ("Stage2", 3, 24, 54, 24),
        ("Stage3", 5, 24, 108, 48),
        ("Stage4", 11, 48, 216, 96),
        ("Stage5", 7, 96, 432, 192),
    ]

    for stage_idx, (stage_name, depth, in_ch, inner_ch, out_ch) in enumerate(stage_configs):
        stage = stages[stage_idx]
        collector.start_section(stage_name)
        if verbose:
            print(f"\n[{stage_name}] Processing {depth} blocks...")

        for block_idx, res_block in enumerate(stage.res_blocks):
            block_name = f"{stage_name}.block{block_idx}"
            bottleneck = res_block.branch2
            if verbose:
                print(f"  --- {block_name} ---")

            # conv_a — 1x1x1 pointwise expand
            input_shape = x.shape
            with collector.time_layer(
                f"{block_name}.conv_a", "QuantizedConv3d (1x1x1)",
                input_shape,
                params=count_parameters(bottleneck.conv_a.weight_q, bottleneck.conv_a.bias_q),
                flops=estimate_conv3d_flops(input_shape, (input_shape[0], inner_ch, *input_shape[2:]), (1, 1, 1)),
                kernel_size=(1, 1, 1),
            ) as timer:
                branch2_x = bottleneck.conv_a.forward(x)
                timer.set_output_shape(branch2_x.shape)

            # norm_a — identity (BN folded into conv_a)
            with collector.time_layer(
                f"{block_name}.norm_a", "Identity (fused BN)",
                branch2_x.shape,
                params=0,
            ) as timer:
                branch2_x = bottleneck.norm_a.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            with collector.time_layer(f"{block_name}.relu_a", "ReLU", branch2_x.shape) as timer:
                branch2_x = relu(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            # conv_b — 3x3x3 depthwise
            stride = 2 if block_idx == 0 else 1
            expected_spatial = (branch2_x.shape[3] // stride, branch2_x.shape[4] // stride)
            input_shape = branch2_x.shape
            with collector.time_layer(
                f"{block_name}.conv_b", "QuantizedConv3d (3x3x3 depthwise)",
                input_shape,
                params=count_parameters(bottleneck.conv_b.weight_q, bottleneck.conv_b.bias_q),
                flops=estimate_conv3d_flops(
                    input_shape,
                    (input_shape[0], inner_ch, input_shape[2], *expected_spatial),
                    (3, 3, 3),
                    groups=inner_ch,
                ),
                kernel_size=(3, 3, 3), stride=(1, stride, stride), groups=inner_ch,
            ) as timer:
                branch2_x = bottleneck.conv_b.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            # norm_b: Sequential([Identity(fused BN), SE_or_Identity])
            norm_b_seq = bottleneck.norm_b
            bn = norm_b_seq._modules["0"]
            se_or_id = norm_b_seq._modules["1"]

            with collector.time_layer(
                f"{block_name}.norm_b.bn", "Identity (fused BN)",
                branch2_x.shape,
                params=0,
            ) as timer:
                branch2_x = bn.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            # SE block (even-indexed blocks only)
            use_se = block_idx % 2 == 0
            if use_se and hasattr(se_or_id, "conv1"):
                se = se_or_id
                input_shape = branch2_x.shape
                with collector.time_layer(
                    f"{block_name}.norm_b.se.squeeze", "AdaptiveAvgPool3d",
                    input_shape,
                    output_size=1,
                ) as timer:
                    se_scale = adaptive_avg_pool3d_forward(branch2_x, 1)
                    timer.set_output_shape(se_scale.shape)
                with collector.time_layer(
                    f"{block_name}.norm_b.se.conv1", "QuantizedConv3d (1x1x1)",
                    se_scale.shape,
                    params=count_parameters(se.conv1.weight_q, se.conv1.bias_q),
                    flops=estimate_conv3d_flops(
                        se_scale.shape,
                        (se_scale.shape[0], se.conv1.out_channels, 1, 1, 1),
                        (1, 1, 1),
                    ),
                ) as timer:
                    se_scale = se.conv1.forward(se_scale)
                    timer.set_output_shape(se_scale.shape)
                with collector.time_layer(
                    f"{block_name}.norm_b.se.relu", "ReLU",
                    se_scale.shape,
                ) as timer:
                    se_scale = relu(se_scale)
                    timer.set_output_shape(se_scale.shape)
                with collector.time_layer(
                    f"{block_name}.norm_b.se.conv2", "QuantizedConv3d (1x1x1)",
                    se_scale.shape,
                    params=count_parameters(se.conv2.weight_q, se.conv2.bias_q),
                    flops=estimate_conv3d_flops(
                        se_scale.shape,
                        (se_scale.shape[0], branch2_x.shape[1], 1, 1, 1),
                        (1, 1, 1),
                    ),
                ) as timer:
                    se_scale = se.conv2.forward(se_scale)
                    timer.set_output_shape(se_scale.shape)
                with collector.time_layer(
                    f"{block_name}.norm_b.se.sigmoid", "Sigmoid",
                    se_scale.shape,
                ) as timer:
                    se_scale = sigmoid(se_scale)
                    timer.set_output_shape(se_scale.shape)
                with collector.time_layer(
                    f"{block_name}.norm_b.se.scale", "Mul (element-wise)",
                    branch2_x.shape,
                ) as timer:
                    branch2_x = branch2_x * se_scale
                    timer.set_output_shape(branch2_x.shape)
            else:
                with collector.time_layer(
                    f"{block_name}.norm_b.identity", "Identity",
                    branch2_x.shape,
                ) as timer:
                    timer.set_output_shape(branch2_x.shape)

            with collector.time_layer(f"{block_name}.silu", "SiLU", branch2_x.shape) as timer:
                branch2_x = silu(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            # conv_c — 1x1x1 project
            input_shape = branch2_x.shape
            with collector.time_layer(
                f"{block_name}.conv_c", "QuantizedConv3d (1x1x1)",
                input_shape,
                params=count_parameters(bottleneck.conv_c.weight_q, bottleneck.conv_c.bias_q),
                flops=estimate_conv3d_flops(input_shape, (input_shape[0], out_ch, *input_shape[2:]), (1, 1, 1)),
                kernel_size=(1, 1, 1),
            ) as timer:
                branch2_x = bottleneck.conv_c.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            # norm_c — identity (BN folded into conv_c)
            with collector.time_layer(
                f"{block_name}.norm_c", "Identity (fused BN)",
                branch2_x.shape,
                params=0,
            ) as timer:
                branch2_x = bottleneck.norm_c.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            # Skip connection
            if res_block.has_branch1:
                input_shape = x.shape
                with collector.time_layer(
                    f"{block_name}.branch1_conv", "QuantizedConv3d (1x1x1)",
                    input_shape,
                    params=count_parameters(res_block.branch1_conv.weight_q, res_block.branch1_conv.bias_q),
                    flops=estimate_conv3d_flops(input_shape, branch2_x.shape, (1, 1, 1)),
                    kernel_size=(1, 1, 1), stride=(1, stride, stride),
                ) as timer:
                    shortcut = res_block.branch1_conv.forward(x)
                    timer.set_output_shape(shortcut.shape)

                if res_block.has_branch1_norm:
                    with collector.time_layer(
                        f"{block_name}.branch1_norm", "Identity (fused BN)",
                        shortcut.shape,
                        params=0,
                    ) as timer:
                        shortcut = res_block.branch1_norm.forward(shortcut)
                        timer.set_output_shape(shortcut.shape)
            else:
                shortcut = x

            with collector.time_layer(f"{block_name}.residual_add", "Add", branch2_x.shape) as timer:
                x = branch2_x + shortcut
                timer.set_output_shape(x.shape)

            with collector.time_layer(f"{block_name}.relu_out", "ReLU", x.shape) as timer:
                x = relu(x)
                timer.set_output_shape(x.shape)

            if verbose and (block_idx == 0 or block_idx == depth - 1):
                print(f"  Block {block_idx}: {x.shape}")

        collector.end_section(stage_name)
        if verbose:
            print(f"  {stage_name} output: {x.shape}")

    # ---------------------------------------------------------------- Head
    collector.start_section("Head")
    if verbose:
        print("\n[Head] Processing...")

    pool = head.pool

    input_shape = x.shape
    with collector.time_layer(
        "Head.pool.pre_conv", "QuantizedConv3d (1x1x1)",
        input_shape,
        params=count_parameters(pool.pre_conv.weight_q, pool.pre_conv.bias_q),
        flops=estimate_conv3d_flops(input_shape, (input_shape[0], 432, *input_shape[2:]), (1, 1, 1)),
        kernel_size=(1, 1, 1),
    ) as timer:
        x = pool.pre_conv.forward(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer(
        "Head.pool.pre_norm", "Identity (fused BN)",
        x.shape,
        params=0,
    ) as timer:
        x = pool.pre_norm.forward(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer("Head.pool.pre_relu", "ReLU", x.shape) as timer:
        x = relu(x)
        timer.set_output_shape(x.shape)

    input_shape = x.shape
    with collector.time_layer(
        "Head.pool.avgpool", "AvgPool3d",
        input_shape,
        kernel_size=(16, 7, 7),
    ) as timer:
        x = avg_pool3d_forward(x, kernel_size=(16, 7, 7))
        timer.set_output_shape(x.shape)
    if verbose:
        print(f"  AvgPool: {input_shape} -> {x.shape}")

    input_shape = x.shape
    with collector.time_layer(
        "Head.pool.post_conv", "QuantizedConv3d (1x1x1)",
        input_shape,
        params=count_parameters(pool.post_conv.weight_q, pool.post_conv.bias_q),
        flops=estimate_conv3d_flops(input_shape, (input_shape[0], 2048, 1, 1, 1), (1, 1, 1)),
        kernel_size=(1, 1, 1),
    ) as timer:
        x = pool.post_conv.forward(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer("Head.pool.post_relu", "ReLU", x.shape) as timer:
        x = relu(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer("Head.dropout", "Dropout", x.shape, p=0.5) as timer:
        x = dropout_forward(x, head.dropout_p, head.training)
        timer.set_output_shape(x.shape)

    x_flat = x.reshape(x.shape[0], -1)
    input_shape = x_flat.shape
    with collector.time_layer(
        "Head.proj", "Linear",
        input_shape,
        params=count_parameters(head._parameters["proj_weight"], head._parameters["proj_bias"]),
        flops=estimate_linear_flops(2048, head.num_classes, x_flat.shape[0]),
        in_features=2048, out_features=head.num_classes,
    ) as timer:
        x = linear_forward(x_flat, head._parameters["proj_weight"], head._parameters["proj_bias"])
        timer.set_output_shape(x.shape)

    collector.end_section("Head")
    if verbose:
        print(f"  Head output: {x.shape}")

    return x


# ---------------------------------------------------------------------------
# Main — mirrors main.py structure
# ---------------------------------------------------------------------------

def main(
    profile: bool = False,
    full_forward: bool = True,
    notes: str = "",
    output_dir: str = "run_stats",
    backend: str = "reference",
    method: Optional[str] = None,
    weights_path: str = "weights/x3d_m_int8.npz",
    num_classes: int = 400,
    verbose: bool = False,
) -> None:
    """
    Build int8 model, set eval mode, and run inference with optional profiling.

    Args:
        profile: Enable detailed profiling and save statistics.
        full_forward: Run full forward pass (False = stem only, for quick tests).
        notes: Additional notes to include in the statistics report.
        output_dir: Directory to save statistics files.
        backend: "reference" or "fpga".
        method: Int8 conv implementation — "slow", "fast", "threaded", "native",
                or None (quantized-subpackage reference kernel).
        weights_path: Path to int8 .npz.
        num_classes: Number of output classes.
        verbose: Print extra loader progress.
    """
    method_label = method or "reference"

    print("=" * 60)
    print("X3D-M Int8 Inference")
    print(f"Conv3d int8 method: {method_label}")
    print(f"Backend: {backend}")
    print("=" * 60)

    wp = Path(weights_path)
    resolved_weights: Optional[Path] = wp if wp.is_file() else None
    if resolved_weights is None:
        print(f"  WARNING: weights file not found at {wp}. "
              f"Using zero-initialized int8 parameters; outputs will be meaningless.")

    model = build_int8_x3d_m(
        num_classes=num_classes,
        weights_path=resolved_weights,
        backend=backend,
        method=method,
        verbose=verbose,
    )
    model.eval()

    x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
    print(f"\nInput shape: {x.shape}")

    if profile:
        collector = StatsCollector("X3D-M-Int8", verbose_log=True)
        collector.start_run(x.shape, notes=notes)

        if full_forward:
            print("\nRunning FULL forward pass with profiling (this may take a while)...")
            logits = run_forward_profiled(model, x, collector, verbose=True)
            print(f"\nOutput shape: {logits.shape}")
            print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
        else:
            print("\nRunning STEM only with profiling (--stem-only)...")
            stem = model.blocks[0]

            collector.start_section("Stem")
            from scratch.ops.activations import relu

            input_shape = x.shape
            with collector.time_layer(
                "Stem.conv.conv_t", "QuantizedConv3d",
                input_shape,
                params=count_parameters(stem.conv.conv_t.weight_q, stem.conv.conv_t.bias_q),
                flops=estimate_conv3d_flops(input_shape, (1, 24, 16, 112, 112), (1, 3, 3), groups=1),
            ) as timer:
                out = stem.conv.conv_t.forward(x)
                timer.set_output_shape(out.shape)
            print(f"  conv_t: {input_shape} -> {out.shape}")

            input_shape = out.shape
            with collector.time_layer(
                "Stem.conv.conv_xy", "QuantizedConv3d (depthwise)",
                input_shape,
                params=count_parameters(stem.conv.conv_xy.weight_q, stem.conv.conv_xy.bias_q),
                flops=estimate_conv3d_flops(input_shape, out.shape, (5, 1, 1), groups=24),
            ) as timer:
                out = stem.conv.conv_xy.forward(out)
                timer.set_output_shape(out.shape)
            print(f"  conv_xy: {input_shape} -> {out.shape}")

            with collector.time_layer(
                "Stem.norm", "Identity (fused BN)",
                out.shape,
                params=0,
            ) as timer:
                out = stem.norm.forward(out)
                timer.set_output_shape(out.shape)

            with collector.time_layer("Stem.relu", "ReLU", out.shape) as timer:
                out = relu(out)
                timer.set_output_shape(out.shape)

            collector.end_section("Stem")
            print(f"\nStem output shape: {out.shape}")
            print(f"Min/Max (stem): {float(out.min()):.4f} / {float(out.max()):.4f}")

        collector.end_run()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        json_path = collector.save(output_dir)
        txt_path = collector.save_text_report(output_dir)

        print(f"\n" + "=" * 60)
        print("STATISTICS SAVED")
        print("=" * 60)
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")

        collector.print_summary()

    else:
        print("\nRunning stem only (no profiling)...")
        stem = model.blocks[0]
        start_time = time.perf_counter()
        out = stem.forward(x)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"Stem output shape: {out.shape}")
        print(f"Stem latency: {elapsed_ms:.2f} ms")
        print(f"Min/Max (stem): {float(out.min()):.4f} / {float(out.max()):.4f}")

        if full_forward:
            print("\nRunning full forward pass (this may take a while)...")
            start_time = time.perf_counter()
            logits = run_forward(model, x)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"Output shape: {logits.shape}")
            print(f"Total latency: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Int8 X3D-M Inference with Profiling")
    parser.add_argument(
        "--profile", "-p",
        action="store_true",
        help="Enable detailed profiling and save statistics",
    )
    parser.add_argument(
        "--stem-only",
        action="store_true",
        help="Profile only the stem block (quick test). Default: full model.",
    )
    parser.add_argument(
        "--notes", "-n",
        type=str,
        default="",
        help="Additional notes to include in the statistics report",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="run_stats",
        help="Directory to save statistics files (default: run_stats)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/x3d_m_int8.npz",
        help="Path to int8 .npz (default: weights/x3d_m_int8.npz)",
    )
    parser.add_argument(
        "--backend",
        choices=["reference", "fpga"],
        default="reference",
        help="Int8 conv backend (default: reference software kernel).",
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default=None,
        choices=list(INT8_METHODS),
        help=(
            "Int8 conv implementation from scratch.ops.conv3d_int8. "
            "When set, overrides the quantized-subpackage reference kernel. "
            '"slow" (pure NumPy loops), '
            '"fast" (vectorised NumPy, default when set), '
            '"threaded" (multi-threaded NumPy), '
            '"native" (C backend via ctypes).'
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the model but do not run a forward pass. Useful for "
             "checking that weights load cleanly.",
    )

    args = parser.parse_args()

    if args.dry_run:
        method_label = args.method or "reference"
        print(f"Building QuantizedX3D_M (method={method_label}) ...")
        wp = Path(args.weights)
        model = build_int8_x3d_m(
            num_classes=args.num_classes,
            weights_path=wp if wp.is_file() else None,
            backend=args.backend,
            method=args.method,
            verbose=args.verbose,
        )
        model.eval()
        print("Dry run: model built successfully.")
    else:
        main(
            profile=args.profile,
            full_forward=not args.stem_only,
            notes=args.notes,
            output_dir=args.output_dir,
            backend=args.backend,
            method=args.method,
            weights_path=args.weights,
            num_classes=args.num_classes,
            verbose=args.verbose,
        )
