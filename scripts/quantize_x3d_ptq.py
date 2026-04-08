"""
Post-Training Quantization (PTQ) for X3D-M to int8.

This script performs static post-training quantization on the pretrained X3D-M
float32 model and exports quantized weights + per-channel scales/zero-points to
a .npz file that can be loaded on the PolarFire SoC. The quantization scheme is
deliberately FPGA-friendly:

    - Weights:      symmetric, per-output-channel int8  (zero_point = 0)
    - Activations:  symmetric, per-tensor int8          (zero_point = 0)
    - Accumulator:  int32
    - BatchNorm:    folded into the preceding Conv3d before quantization
    - Bias:         kept as int32, scale = input_scale * weight_scale[c]

Symmetric quantization eliminates zero-point subtraction from the weight path,
which is the single biggest simplification for an FPGA MAC array. Per-channel
weight scales are essentially free in hardware (one scale per output channel,
applied at requantization time) but recover most of the accuracy lost to PTQ.

This script MUST be run on a machine with PyTorch installed (e.g. a MacBook).
The resulting .npz is portable and can be consumed by the scratch library on
the RISC-V target with no PyTorch dependency.

Pipeline
--------
1. Load pretrained float32 X3D-M from PyTorchVideo hub (or a local .pth).
2. Fold every BatchNorm3d into the preceding Conv3d (weights and bias rewritten
   in-place). After folding, BN layers become identity and are skipped.
3. Run a calibration loop over a small set of representative video clips
   (default: random tensors; pass --calib-dir to use real clips) to collect
   per-tensor activation statistics (absolute max) at the input of every
   Conv3d/Linear layer.
4. Compute scales:
       weight_scale[c] = max(|W[c]|) / 127        (per output channel)
       act_scale       = max(|A|)    / 127        (per tensor, from calibration)
5. Quantize weights to int8, quantize biases to int32 using
   bias_scale[c] = input_scale * weight_scale[c].
6. Save everything to a .npz with the following keys per layer `L`:
       L.weight_q     : int8,  shape (out_c, in_c/groups, kT, kH, kW)
       L.weight_scale : float32, shape (out_c,)
       L.bias_q       : int32, shape (out_c,)  (optional)
       L.input_scale  : float32, scalar
       L.output_scale : float32, scalar
   Plus global metadata under the `__meta__` key.

Usage
-----
    # Quantize from the PyTorchVideo hub checkpoint (default)
    python scripts/quantize_x3d_ptq.py \\
        -o weights/x3d_m_int8.npz

    # Quantize from a local float32 .pth file
    python scripts/quantize_x3d_ptq.py \\
        -i weights/x3d_m_kinetics400.pth \\
        -o weights/x3d_m_int8.npz \\
        --num-calib-batches 64

    # Use real calibration clips (directory of .npy files shaped (3,16,224,224))
    python scripts/quantize_x3d_ptq.py \\
        --calib-dir data/kinetics_calib \\
        -o weights/x3d_m_int8.npz

Notes
-----
- On an M3 Max with 48 GB RAM, 128 calibration batches of a single clip each
  takes roughly 1-2 minutes on CPU. MPS acceleration is used automatically if
  available.
- SiLU activations are NOT quantized to a lookup table here; the exported file
  assumes the SoC runtime will evaluate SiLU in float on the dequantized
  activations, then requantize. If you want a pure-int8 SiLU LUT, generate it
  separately from the exported activation scales.
- This is *post-training* quantization only. If accuracy on Kinetics-400 drops
  more than ~2% top-1, consider quantization-aware training (QAT) as a
  follow-up step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv: "nn.Conv3d", bn: "nn.BatchNorm3d") -> None:
    """
    Fold a BatchNorm3d into the preceding Conv3d in-place.

    After folding:
        W'[c] = W[c] * gamma[c] / sqrt(var[c] + eps)
        b'[c] = (b[c] - mean[c]) * gamma[c] / sqrt(var[c] + eps) + beta[c]

    The BN layer should subsequently be replaced with nn.Identity() so it does
    not apply a second normalization at inference time.
    """
    assert isinstance(conv, nn.Conv3d), f"Expected Conv3d, got {type(conv)}"
    assert isinstance(bn, nn.BatchNorm3d), f"Expected BatchNorm3d, got {type(bn)}"

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    inv_std = torch.rsqrt(var + eps)          # (C,)
    scale = gamma * inv_std                   # (C,)

    # Reshape scale for broadcasting over (out_c, in_c, kT, kH, kW)
    w = conv.weight.detach()
    w_folded = w * scale.reshape(-1, 1, 1, 1, 1)

    if conv.bias is None:
        b_folded = (-mean) * scale + beta
        conv.bias = nn.Parameter(b_folded)
    else:
        b = conv.bias.detach()
        b_folded = (b - mean) * scale + beta
        conv.bias.data.copy_(b_folded)

    conv.weight.data.copy_(w_folded)


def fold_all_bn(model: "nn.Module") -> None:
    """
    Walk the model and fold every BatchNorm3d into its preceding Conv3d.

    Uses a simple pattern-match pass: for each parent module, iterate its
    named children in order and when we see (Conv3d, BatchNorm3d) adjacent we
    fold and replace the BN with Identity. This handles the X3D-M structure
    because every BN immediately follows a Conv3d in the module ordering
    within its parent.
    """
    for parent in model.modules():
        children = list(parent.named_children())
        for i in range(len(children) - 1):
            name_a, mod_a = children[i]
            name_b, mod_b = children[i + 1]
            if isinstance(mod_a, nn.Conv3d) and isinstance(mod_b, nn.BatchNorm3d):
                fold_bn_into_conv(mod_a, mod_b)
                setattr(parent, name_b, nn.Identity())


# ---------------------------------------------------------------------------
# Activation calibration
# ---------------------------------------------------------------------------

class ActivationObserver:
    """
    Hook-based running absolute-max observer for a single tensor location.

    We use absolute max (not percentile) for simplicity and because symmetric
    per-tensor quantization only needs |max|. For production you may want to
    replace this with a 99.99 percentile to reduce outlier sensitivity.
    """

    def __init__(self) -> None:
        self.abs_max: float = 0.0
        self.num_batches: int = 0

    def update(self, x: "torch.Tensor") -> None:
        val = float(x.detach().abs().max().item())
        if val > self.abs_max:
            self.abs_max = val
        self.num_batches += 1

    @property
    def scale(self) -> float:
        """int8 symmetric scale so that quantize(x) lies in [-127, 127]."""
        if self.abs_max == 0.0:
            return 1.0
        return self.abs_max / 127.0


def attach_observers(
    model: "nn.Module",
) -> Tuple[Dict[str, ActivationObserver], Dict[str, ActivationObserver], List]:
    """
    Attach forward pre-hooks (input) and forward hooks (output) to every
    Conv3d and Linear in the model.

    Returns
    -------
    input_obs : dict[str, ActivationObserver]
        Observers for the input of each quantized layer.
    output_obs : dict[str, ActivationObserver]
        Observers for the output of each quantized layer.
    handles : list
        Hook handles (keep a reference so they aren't GC'd during calibration).
    """
    input_obs: Dict[str, ActivationObserver] = {}
    output_obs: Dict[str, ActivationObserver] = {}
    handles = []

    for name, module in model.named_modules():
        # NOTE: in Phase 1 of the FPGA project, only Conv3d is offloaded to
        # the accelerator. The final Linear in the head (blocks.5.proj) is
        # left in float32 and handled by the CPU, so we do not quantize or
        # observe it here.
        if not isinstance(module, nn.Conv3d):
            continue
        input_obs[name] = ActivationObserver()
        output_obs[name] = ActivationObserver()

        def _pre_hook(_mod, inputs, _name=name):
            if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                input_obs[_name].update(inputs[0])

        def _post_hook(_mod, _inputs, output, _name=name):
            if isinstance(output, torch.Tensor):
                output_obs[_name].update(output)

        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))

    return input_obs, output_obs, handles


def iter_calibration_batches(
    calib_dir: Path | None,
    num_batches: int,
    device: "torch.device",
):
    """
    Yield calibration tensors shaped (1, 3, 16, 224, 224).

    If `calib_dir` is provided, loads .npy files from it (each expected to be
    already preprocessed into (3,16,224,224) float32, normalized the same way
    the training pipeline normalized). Otherwise yields random tensors sampled
    from a standard-normal distribution, which is adequate for capturing rough
    activation magnitudes on a pretrained model but not ideal for production.
    """
    if calib_dir is not None:
        files = sorted(calib_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in {calib_dir}")
        for i, f in enumerate(files[:num_batches]):
            arr = np.load(f).astype(np.float32)
            assert arr.shape == (3, 16, 224, 224), f"Bad shape {arr.shape} in {f}"
            yield torch.from_numpy(arr).unsqueeze(0).to(device)
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        for _ in range(num_batches):
            x = torch.randn(1, 3, 16, 224, 224, generator=g)
            yield x.to(device)


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------

def quantize_weight_per_channel(w: "torch.Tensor") -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-output-channel int8 quantization.

    Parameters
    ----------
    w : Tensor, shape (out_c, in_c/groups, kT, kH, kW) for Conv3d
        or (out, in) for Linear.

    Returns
    -------
    w_q : int8 ndarray, same shape as w
    scales : float32 ndarray, shape (out_c,)
    """
    w_np = w.detach().cpu().float().numpy()
    out_c = w_np.shape[0]
    flat = w_np.reshape(out_c, -1)
    abs_max = np.max(np.abs(flat), axis=1)             # (out_c,)
    # Avoid divide-by-zero for dead channels
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = (abs_max / 127.0).astype(np.float32)      # (out_c,)
    # Broadcast scales across non-channel dims
    bcast_shape = (out_c,) + (1,) * (w_np.ndim - 1)
    w_q = np.round(w_np / scales.reshape(bcast_shape))
    w_q = np.clip(w_q, -127, 127).astype(np.int8)
    return w_q, scales


def quantize_bias(
    b: "torch.Tensor", input_scale: float, weight_scales: np.ndarray
) -> np.ndarray:
    """
    Quantize a bias vector to int32 with scale = input_scale * weight_scale[c].

    int32 biases can be added directly to the int32 accumulator of the
    convolution without any additional rescaling, which is the standard
    pattern for hardware int8 GEMM/conv units.
    """
    b_np = b.detach().cpu().float().numpy()
    bias_scale = input_scale * weight_scales                # (out_c,)
    bias_scale = np.where(bias_scale == 0, 1.0, bias_scale)
    b_q = np.round(b_np / bias_scale)
    b_q = np.clip(b_q, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return b_q.astype(np.int32)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_quantized(
    model: "nn.Module",
    input_obs: Dict[str, ActivationObserver],
    output_obs: Dict[str, ActivationObserver],
    output_path: Path,
) -> None:
    """
    Walk the model, quantize every Conv3d/Linear layer, and save to .npz.

    The output .npz is flat (no nesting). Keys follow the scheme:
        <layer_name>.weight_q
        <layer_name>.weight_scale
        <layer_name>.bias_q             (only if bias is not None)
        <layer_name>.input_scale
        <layer_name>.output_scale
    """
    out: Dict[str, np.ndarray] = {}
    n_layers = 0

    # Dump the Head's final Linear (blocks.5.proj) as FLOAT32. Under the
    # Phase-1 "only convolutions on the FPGA" plan, this layer is not
    # quantized — it stays float32 on the CPU — but its weights still need
    # to travel with the int8 .npz so the quantized model can load them.
    # Scratch-side keys are `blocks.5.proj_weight` / `blocks.5.proj_bias`
    # (see scripts/convert_pytorch_weights_to_numpy.py).
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith("proj"):
            parent_path = name.rsplit(".", 1)[0]  # e.g. "blocks.5"
            out[f"{parent_path}.proj_weight"] = (
                module.weight.detach().cpu().float().numpy().astype(np.float32)
            )
            if module.bias is not None:
                out[f"{parent_path}.proj_bias"] = (
                    module.bias.detach().cpu().float().numpy().astype(np.float32)
                )

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv3d):
            continue
        in_scale = np.float32(input_obs[name].scale)
        out_scale = np.float32(output_obs[name].scale)

        w_q, w_scales = quantize_weight_per_channel(module.weight)
        out[f"{name}.weight_q"] = w_q
        out[f"{name}.weight_scale"] = w_scales
        out[f"{name}.input_scale"] = in_scale
        out[f"{name}.output_scale"] = out_scale

        if module.bias is not None:
            out[f"{name}.bias_q"] = quantize_bias(module.bias, float(in_scale), w_scales)

        n_layers += 1

    # Metadata so the SoC-side loader can sanity-check the file.
    out["__meta__.num_quantized_layers"] = np.int32(n_layers)
    out["__meta__.weight_scheme"] = np.array("symmetric_per_channel_int8", dtype="S64")
    out["__meta__.act_scheme"] = np.array("symmetric_per_tensor_int8", dtype="S64")
    out["__meta__.bias_scheme"] = np.array("int32_input_scale_x_weight_scale", dtype="S64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    print(f"Saved {n_layers} quantized layers to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(input_path: Path | None) -> "nn.Module":
    """Load a float32 X3D-M either from the hub or a local .pth state_dict."""
    if input_path is None:
        print("Loading pretrained X3D-M from PyTorchVideo hub ...")
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
    else:
        print(f"Loading X3D-M architecture from hub and weights from {input_path} ...")
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=False)
        try:
            sd = torch.load(input_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(input_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    model.eval()
    return model


def pick_device() -> "torch.device":
    """Prefer MPS on Apple Silicon, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-training int8 quantization for X3D-M (FPGA-friendly scheme)."
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=None,
        help="Optional input float32 .pth/.pt checkpoint. If omitted, load from hub.",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("weights/x3d_m_int8.npz"),
        help="Output .npz path for quantized weights.",
    )
    parser.add_argument(
        "--calib-dir", type=Path, default=None,
        help="Directory of .npy calibration clips shaped (3,16,224,224). "
             "If omitted, random tensors are used (quick but lower quality).",
    )
    parser.add_argument(
        "--num-calib-batches", type=int, default=128,
        help="Number of calibration batches to run (default: 128).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: cpu, mps, or cuda. Default: auto-detect.",
    )
    args = parser.parse_args()

    if torch is None:
        print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        return 1

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")

    # 1. Load pretrained float32 model
    model = load_model(args.input).to(device)

    # 2. Fold BatchNorm into preceding Conv3d. This MUST happen before
    #    calibration so the activation statistics reflect the folded graph
    #    actually deployed on the SoC.
    print("Folding BatchNorm into Conv3d ...")
    fold_all_bn(model)
    model.eval()

    # 3. Attach observers and run calibration
    print("Attaching activation observers ...")
    input_obs, output_obs, handles = attach_observers(model)

    print(f"Running calibration on {args.num_calib_batches} batches ...")
    with torch.no_grad():
        for i, x in enumerate(iter_calibration_batches(args.calib_dir, args.num_calib_batches, device)):
            _ = model(x)
            if (i + 1) % 16 == 0:
                print(f"  calibrated {i + 1}/{args.num_calib_batches}")

    for h in handles:
        h.remove()

    # 4. Quantize and export
    print("Quantizing weights and exporting ...")
    export_quantized(model, input_obs, output_obs, args.output)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
