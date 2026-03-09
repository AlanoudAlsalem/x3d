"""
X3D-M built from scratch (no PyTorch) for RISC-V SoC compatibility.

Uses the modular scratch implementation in the `scratch` package. All tensors
are NumPy arrays; layout is (B, C, T, H, W) for 3D data.

This version includes comprehensive profiling and statistics collection for
performance analysis across different platforms (macOS, PolarFire SoC, etc.).

Example (random init):
    from main import build_x3d_m, run_forward
    import numpy as np
    model = build_x3d_m(num_classes=400)
    model.eval()
    x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
    logits = run_forward(model, x)

Example (with pretrained weights on SoC):
    model = build_x3d_m(num_classes=400, weights_path="weights/x3d_m_kinetics400.npz")
    model.eval()

Example (with profiling):
    python main.py --profile --notes "Testing on MacBook Pro"
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scratch.models.x3d_m import X3D_M
from scratch.load_weights import load_pretrained_numpy, load_pretrained_numpy_if_available
from scratch.stats import (
    StatsCollector,
    estimate_conv3d_flops,
    estimate_linear_flops,
    count_parameters,
)


def build_x3d_m(
    num_classes: int = 400,
    weights_path: Optional[Union[str, Path]] = None,
    *,
    strict_weights: bool = True,
) -> X3D_M:
    """
    Build X3D-M model (optionally load pretrained weights). No PyTorch.

    Weights file must be a .npz produced by scripts/convert_pytorch_weights_to_numpy.py
    on a machine with PyTorch, then copied to the SoC.

    Args:
        num_classes: Number of output classes (default 400 for Kinetics).
        weights_path: Optional path to .npz file; if set, load into model.
        strict_weights: If True, raise when weight file is missing a parameter.

    Returns:
        X3D_M instance. Call model.eval() for inference.
    """
    model = X3D_M(num_classes=num_classes)
    if weights_path is not None:
        load_pretrained_numpy(model, weights_path, strict=strict_weights)
    return model


def run_forward(model: X3D_M, x: np.ndarray) -> np.ndarray:
    """
    Run one forward pass.

    Args:
        model: X3D_M from build_x3d_m().
        x: Input array (B, 3, 16, 224, 224), dtype float32.

    Returns:
        Logits array (B, num_classes).
    """
    return model.forward(x)


def run_forward_profiled(
    model: X3D_M,
    x: np.ndarray,
    collector: StatsCollector,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run forward pass with detailed per-layer profiling.

    Args:
        model: X3D_M model instance.
        x: Input array (B, 3, 16, 224, 224).
        collector: StatsCollector instance.
        verbose: Print progress during inference.

    Returns:
        Logits array (B, num_classes).
    """
    from scratch.ops.activations import relu, silu, sigmoid
    from scratch.ops.pooling import avg_pool3d_forward, adaptive_avg_pool3d_forward
    from scratch.ops.linear import linear_forward
    from scratch.ops.dropout import dropout_forward

    if verbose:
        print("\n" + "=" * 60)
        print("PROFILED FORWARD PASS")
        print("=" * 60)

    stem = model.blocks[0]
    stages = [model.blocks[i] for i in range(1, 5)]
    head = model.blocks[5]

    collector.start_section("Stem")
    if verbose:
        print("\n[Stem] Processing...")

    input_shape = x.shape
    with collector.time_layer(
        "Stem.conv.conv_t", "Conv3d",
        input_shape,
        params=count_parameters(stem.conv.conv_t._parameters["weight"]),
        flops=estimate_conv3d_flops(input_shape, (1, 24, 16, 112, 112), (1, 3, 3), groups=1),
        kernel_size=(1, 3, 3), stride=(1, 2, 2), groups=1,
    ) as timer:
        x = stem.conv.conv_t.forward(x)
        timer.set_output_shape(x.shape)
    if verbose:
        print(f"  conv_t: {input_shape} -> {x.shape}")

    input_shape = x.shape
    with collector.time_layer(
        "Stem.conv.conv_xy", "Conv3d (depthwise)",
        input_shape,
        params=count_parameters(stem.conv.conv_xy._parameters["weight"]),
        flops=estimate_conv3d_flops(input_shape, x.shape, (5, 1, 1), groups=24),
        kernel_size=(5, 1, 1), stride=1, groups=24,
    ) as timer:
        x = stem.conv.conv_xy.forward(x)
        timer.set_output_shape(x.shape)
    if verbose:
        print(f"  conv_xy: {input_shape} -> {x.shape}")

    input_shape = x.shape
    with collector.time_layer(
        "Stem.norm", "BatchNorm3d",
        input_shape,
        params=count_parameters(
            stem.norm._parameters["weight"],
            stem.norm._parameters["bias"]
        ),
    ) as timer:
        x = stem.norm.forward(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer("Stem.relu", "ReLU", x.shape) as timer:
        x = relu(x)
        timer.set_output_shape(x.shape)

    collector.end_section("Stem")
    if verbose:
        print(f"  Stem output: {x.shape}")

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

            input_shape = x.shape
            with collector.time_layer(
                f"{block_name}.conv_a", "Conv3d (1x1x1)",
                input_shape,
                params=count_parameters(bottleneck.conv_a._parameters["weight"]),
                flops=estimate_conv3d_flops(input_shape, (input_shape[0], inner_ch, *input_shape[2:]), (1, 1, 1)),
                kernel_size=(1, 1, 1),
            ) as timer:
                branch2_x = bottleneck.conv_a.forward(x)
                timer.set_output_shape(branch2_x.shape)

            with collector.time_layer(
                f"{block_name}.norm_a", "BatchNorm3d",
                branch2_x.shape,
                params=count_parameters(
                    bottleneck.norm_a._parameters["weight"],
                    bottleneck.norm_a._parameters["bias"]
                ),
            ) as timer:
                branch2_x = bottleneck.norm_a.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            with collector.time_layer(f"{block_name}.relu_a", "ReLU", branch2_x.shape) as timer:
                branch2_x = relu(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            stride = 2 if block_idx == 0 else 1
            expected_spatial = (branch2_x.shape[3] // stride, branch2_x.shape[4] // stride)
            input_shape = branch2_x.shape
            with collector.time_layer(
                f"{block_name}.conv_b", "Conv3d (3x3x3 depthwise)",
                input_shape,
                params=count_parameters(bottleneck.conv_b._parameters["weight"]),
                flops=estimate_conv3d_flops(
                    input_shape,
                    (input_shape[0], inner_ch, input_shape[2], *expected_spatial),
                    (3, 3, 3),
                    groups=inner_ch
                ),
                kernel_size=(3, 3, 3), stride=(1, stride, stride), groups=inner_ch,
            ) as timer:
                branch2_x = bottleneck.conv_b.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            norm_b_seq = bottleneck.norm_b
            bn = norm_b_seq._modules["0"]
            se_or_id = norm_b_seq._modules["1"]

            with collector.time_layer(
                f"{block_name}.norm_b.bn", "BatchNorm3d",
                branch2_x.shape,
                params=count_parameters(bn._parameters["weight"], bn._parameters["bias"]),
            ) as timer:
                branch2_x = bn.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

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
                    f"{block_name}.norm_b.se.conv1", "Conv3d (1x1x1)",
                    se_scale.shape,
                    params=count_parameters(
                        se.conv1._parameters["weight"],
                        se.conv1._parameters.get("bias"),
                    ),
                    flops=estimate_conv3d_flops(
                        se_scale.shape,
                        (se_scale.shape[0], se.conv2._parameters["weight"].shape[1], 1, 1, 1),
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
                    f"{block_name}.norm_b.se.conv2", "Conv3d (1x1x1)",
                    se_scale.shape,
                    params=count_parameters(
                        se.conv2._parameters["weight"],
                        se.conv2._parameters.get("bias"),
                    ),
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

            input_shape = branch2_x.shape
            with collector.time_layer(
                f"{block_name}.conv_c", "Conv3d (1x1x1)",
                input_shape,
                params=count_parameters(bottleneck.conv_c._parameters["weight"]),
                flops=estimate_conv3d_flops(input_shape, (input_shape[0], out_ch, *input_shape[2:]), (1, 1, 1)),
                kernel_size=(1, 1, 1),
            ) as timer:
                branch2_x = bottleneck.conv_c.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            with collector.time_layer(
                f"{block_name}.norm_c", "BatchNorm3d",
                branch2_x.shape,
                params=count_parameters(
                    bottleneck.norm_c._parameters["weight"],
                    bottleneck.norm_c._parameters["bias"]
                ),
            ) as timer:
                branch2_x = bottleneck.norm_c.forward(branch2_x)
                timer.set_output_shape(branch2_x.shape)

            if res_block.has_branch1:
                input_shape = x.shape
                with collector.time_layer(
                    f"{block_name}.branch1_conv", "Conv3d (1x1x1)",
                    input_shape,
                    params=count_parameters(res_block.branch1_conv._parameters["weight"]),
                    flops=estimate_conv3d_flops(input_shape, branch2_x.shape, (1, 1, 1)),
                    kernel_size=(1, 1, 1), stride=(1, stride, stride),
                ) as timer:
                    shortcut = res_block.branch1_conv.forward(x)
                    timer.set_output_shape(shortcut.shape)

                if res_block.has_branch1_norm:
                    with collector.time_layer(
                        f"{block_name}.branch1_norm", "BatchNorm3d",
                        shortcut.shape,
                        params=count_parameters(
                            res_block.branch1_norm._parameters["weight"],
                            res_block.branch1_norm._parameters["bias"]
                        ),
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

    collector.start_section("Head")
    if verbose:
        print("\n[Head] Processing...")

    pool = head.pool

    input_shape = x.shape
    with collector.time_layer(
        "Head.pool.pre_conv", "Conv3d (1x1x1)",
        input_shape,
        params=count_parameters(pool.pre_conv._parameters["weight"]),
        flops=estimate_conv3d_flops(input_shape, (input_shape[0], 432, *input_shape[2:]), (1, 1, 1)),
        kernel_size=(1, 1, 1),
    ) as timer:
        x = pool.pre_conv.forward(x)
        timer.set_output_shape(x.shape)

    with collector.time_layer(
        "Head.pool.pre_norm", "BatchNorm3d",
        x.shape,
        params=count_parameters(pool.pre_norm._parameters["weight"], pool.pre_norm._parameters["bias"]),
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
        "Head.pool.post_conv", "Conv3d (1x1x1)",
        input_shape,
        params=count_parameters(pool.post_conv._parameters["weight"]),
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


def main(
    profile: bool = False,
    full_forward: bool = True,
    notes: str = "",
    output_dir: str = "run_stats",
) -> None:
    """
    Build model, set eval mode, and run inference with optional profiling.

    Args:
        profile: Enable detailed profiling and save statistics.
        full_forward: Run full forward pass (False = stem only, for quick tests).
        notes: Additional notes to include in the statistics report.
        output_dir: Directory to save statistics files.
    """
    print("=" * 60)
    print("X3D-M Inference")
    print("=" * 60)

    weights_path = "weights/x3d_m_kinetics400.npz"
    model = build_x3d_m(num_classes=400, weights_path=weights_path)
    load_pretrained_numpy_if_available(model, weights_path, verbose=True)
    model.eval()

    x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
    print(f"\nInput shape: {x.shape}")

    if profile:
        collector = StatsCollector("X3D-M", verbose_log=True)
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
                "Stem.conv.conv_t", "Conv3d",
                input_shape,
                params=count_parameters(stem.conv.conv_t._parameters["weight"]),
                flops=estimate_conv3d_flops(input_shape, (1, 24, 16, 112, 112), (1, 3, 3), groups=1),
            ) as timer:
                out = stem.conv.conv_t.forward(x)
                timer.set_output_shape(out.shape)
            print(f"  conv_t: {input_shape} -> {out.shape}")

            input_shape = out.shape
            with collector.time_layer(
                "Stem.conv.conv_xy", "Conv3d (depthwise)",
                input_shape,
                params=count_parameters(stem.conv.conv_xy._parameters["weight"]),
                flops=estimate_conv3d_flops(input_shape, out.shape, (5, 1, 1), groups=24),
            ) as timer:
                out = stem.conv.conv_xy.forward(out)
                timer.set_output_shape(out.shape)
            print(f"  conv_xy: {input_shape} -> {out.shape}")

            with collector.time_layer(
                "Stem.norm", "BatchNorm3d",
                out.shape,
                params=count_parameters(stem.norm._parameters["weight"], stem.norm._parameters["bias"]),
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
    parser = argparse.ArgumentParser(description="X3D-M Inference with Profiling")
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

    args = parser.parse_args()
    main(
        profile=args.profile,
        full_forward=not args.stem_only,
        notes=args.notes,
        output_dir=args.output_dir,
    )
