"""
Quantization-Aware Training (QAT) for X3D-M to int8.

This script performs quantization-aware fine-tuning on the pretrained X3D-M
model, producing int8 weights that are significantly more accurate than
post-training quantization (PTQ). QAT inserts "fake quantization" nodes into
the forward pass during training so the model learns to be robust to
quantization noise, then exports the final quantized weights.

The quantization scheme matches the existing PTQ pipeline exactly:
    - Weights:      symmetric, per-output-channel int8  (zero_point = 0)
    - Activations:  symmetric, per-tensor int8          (zero_point = 0)
    - Accumulator:  int32
    - BatchNorm:    folded into the preceding Conv3d BEFORE QAT begins
    - Bias:         kept as int32, scale = input_scale * weight_scale[c]

Strategy
--------
1. Load pretrained float32 X3D-M from PyTorchVideo hub.
2. Fold ALL BatchNorm3d layers into their preceding Conv3d. After folding,
   BN layers become nn.Identity() and the conv weights/biases absorb the
   normalization. This is done BEFORE training starts.
3. Insert FakeQuantize nodes on every Conv3d: one on the input (per-tensor)
   and one on the weight (per-output-channel). These simulate int8
   quantization noise while allowing gradients to flow via straight-through
   estimator (STE).
4. Fine-tune the BN-folded, fake-quantized model on Kinetics-400 data
   with a low learning rate (default 1e-5) for 20 epochs.
5. Export the final weights using the exact same .npz format as the PTQ
   script (quantize_x3d_ptq.py), so the SoC-side loader works unchanged.

This script MUST be run on a machine with PyTorch and a Kinetics-400 dataset
(or a subset thereof). The resulting .npz is portable and can be consumed by
the scratch library on the RISC-V target with no PyTorch dependency.

Data Preparation
----------------
Download Kinetics-400 from one of these sources:

  1. Academic Torrents / original Google links:
     https://github.com/cvdfoundation/kinetics-dataset

  2. Smaller "mini" subsets for faster QAT:
     https://github.com/deepmind/kinetics-i3d (sample data)

Organize the data as follows:

    data/kinetics400/
        train/
            abseiling/
                video001.mp4
                video002.mp4
                ...
            air_drumming/
                ...
            ...  (400 class directories)
        val/
            abseiling/
                ...
            ...

Each subdirectory name is the action class label. The script uses the
directory structure to assign integer labels automatically (sorted
alphabetically, 0-indexed).

Alternatively, you can point --train-dir at ANY directory of class-organized
video files, even a small subset of 10-50 classes, for quick experiments.

Usage
-----
    # Standard QAT with 20 epochs (recommended)
    python scripts/quantize_x3d_qat.py \\
        --train-dir data/kinetics400/train \\
        --val-dir data/kinetics400/val \\
        -o weights/x3d_m_int8_qat.npz

    # Quick QAT run with fewer epochs for testing
    python scripts/quantize_x3d_qat.py \\
        --train-dir data/kinetics400/train \\
        --val-dir data/kinetics400/val \\
        --epochs 5 --lr 1e-5 \\
        -o weights/x3d_m_int8_qat.npz

    # Use a small subset of classes
    python scripts/quantize_x3d_qat.py \\
        --train-dir data/kinetics_mini/train \\
        --epochs 10 \\
        -o weights/x3d_m_int8_qat.npz

    # Resume from a QAT checkpoint
    python scripts/quantize_x3d_qat.py \\
        --train-dir data/kinetics400/train \\
        --resume checkpoints/qat_epoch_10.pth \\
        -o weights/x3d_m_int8_qat.npz

Notes
-----
- On an M3 Max MacBook, expect ~2-4 minutes per epoch with a small subset
  (1000 clips), or ~1-2 hours per epoch on the full Kinetics-400 train set.
- MPS (Apple Silicon GPU) is used automatically when available.
- A learning rate of 1e-5 with cosine annealing works well; higher rates
  risk diverging since the model is already pretrained.
- The script saves QAT checkpoints every 5 epochs so you can resume if
  interrupted.
- For best results, use at least 10-20 epochs on a reasonably sized subset
  (10k+ clips) of Kinetics-400.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    nn = None

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================================
# Video Dataset
# ============================================================================

class Kinetics400Dataset(Dataset):
    """
    PyTorch Dataset for Kinetics-400 (or any class-organized video directory).

    Expected directory structure:
        root/
            class_name_1/
                video001.mp4
                video002.avi
                ...
            class_name_2/
                ...

    Each video is decoded on-the-fly using OpenCV. We uniformly sample
    `num_frames` frames from the video, resize to `short_side` on the shorter
    edge, then take a center crop of `crop_size x crop_size`.

    The output tensor is (C, T, H, W) = (3, 16, 224, 224), float32,
    normalized to roughly zero-mean unit-variance using Kinetics mean/std.
    """

    # Kinetics-400 normalization (same as PyTorchVideo's default)
    MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        root: str | Path,
        num_frames: int = 16,
        short_side: int = 256,
        crop_size: int = 224,
        is_train: bool = True,
        max_clips_per_class: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.num_frames = num_frames
        self.short_side = short_side
        self.crop_size = crop_size
        self.is_train = is_train

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Discover classes (sorted for deterministic label assignment)
        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not class_dirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {self.root}. "
                f"Expected structure: {self.root}/<class_name>/<video_files>"
            )

        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.num_classes = len(self.class_to_idx)

        # Collect all video file paths
        self.samples: List[Tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            videos = sorted([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in self.VIDEO_EXTENSIONS
            ])
            if max_clips_per_class > 0:
                videos = videos[:max_clips_per_class]
            for v in videos:
                self.samples.append((v, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No video files found in {self.root}. "
                f"Supported formats: {self.VIDEO_EXTENSIONS}"
            )

        print(f"  Dataset: {len(self.samples)} videos, "
              f"{self.num_classes} classes from {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_frames(self, path: Path) -> Optional[np.ndarray]:
        """
        Load and uniformly sample `num_frames` frames from a video file.
        Returns (T, H, W, 3) uint8 array, or None if the video can't be read.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        # Set a read timeout to avoid hanging on corrupt files.
        # OpenCV's FFMPEG backend can stall for 15+ minutes on partial files.
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)   # 5s open timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)    # 5s read timeout

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Try reading a few frames directly if metadata is missing.
            # Cap at 300 frames to avoid infinite loops on broken streams.
            frames = []
            for _ in range(300):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(frames) < self.num_frames:
                return None
            total_frames = len(frames)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            return np.stack([frames[i] for i in indices])

        # Sanity check: skip absurdly long or zero-length videos
        if total_frames > 100000:
            cap.release()
            return None

        # Uniform temporal sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return np.stack(frames)  # (T, H, W, 3)

    def _spatial_transform(self, frames: np.ndarray) -> np.ndarray:
        """
        Resize + crop frames. Input: (T, H, W, 3) uint8.
        Output: (T, crop_size, crop_size, 3) uint8.
        """
        T, H, W, C = frames.shape

        # Resize so shorter side = self.short_side
        if H < W:
            new_h = self.short_side
            new_w = int(W * self.short_side / H)
        else:
            new_w = self.short_side
            new_h = int(H * self.short_side / W)

        resized = np.stack([
            cv2.resize(frames[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            for t in range(T)
        ])

        _, rH, rW, _ = resized.shape

        if self.is_train:
            # Random crop during training
            top = np.random.randint(0, rH - self.crop_size + 1)
            left = np.random.randint(0, rW - self.crop_size + 1)
        else:
            # Center crop during validation
            top = (rH - self.crop_size) // 2
            left = (rW - self.crop_size) // 2

        cropped = resized[:, top:top + self.crop_size, left:left + self.crop_size, :]
        return cropped

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Try up to 5 different videos before giving up (handles corrupt files)
        for attempt in range(5):
            try:
                idx = index if attempt == 0 else np.random.randint(0, len(self.samples))
                path, label = self.samples[idx]

                frames = self._load_video_frames(path)
                if frames is None:
                    continue

                # Spatial transforms
                frames = self._spatial_transform(frames)  # (T, H, W, 3) uint8

                # Random horizontal flip during training
                if self.is_train and np.random.rand() < 0.5:
                    frames = frames[:, :, ::-1, :].copy()

                # Normalize: uint8 [0,255] -> float32 [0,1] -> (x - mean) / std
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - self.MEAN) / self.STD

                # (T, H, W, C) -> (C, T, H, W) for PyTorch
                frames = np.transpose(frames, (3, 0, 1, 2))

                return torch.from_numpy(frames.copy()), label
            except Exception:
                continue

        # All attempts failed — return a zero tensor with label 0
        dummy = torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)
        return dummy, 0


# ============================================================================
# BatchNorm Folding (reused from PTQ script)
# ============================================================================

def fold_bn_into_conv(conv: "nn.Conv3d", bn: "nn.BatchNorm3d") -> None:
    """
    Fold a BatchNorm3d into the preceding Conv3d in-place.

    After folding:
        W'[c] = W[c] * gamma[c] / sqrt(var[c] + eps)
        b'[c] = (b[c] - mean[c]) * gamma[c] / sqrt(var[c] + eps) + beta[c]
    """
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    inv_std = torch.rsqrt(var + eps)
    scale = gamma * inv_std

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


def fold_all_bn(model: "nn.Module") -> int:
    """
    Walk the model and fold every adjacent (Conv3d, BatchNorm3d) pair.

    This uses the SAME logic as the PTQ script (quantize_x3d_ptq.py) so
    that the exported .npz is identical in structure. Specifically:

    FOLDED (adjacent Conv3d + BatchNorm3d within the same parent):
      - conv_a + norm_a  (in BottleneckBlock)
      - conv_c + norm_c  (in BottleneckBlock)
      - branch1_conv + branch1_norm  (in ResBlock, when present)
      - pre_conv + pre_norm  (in ProjectedPool / Head)

    NOT FOLDED (non-adjacent or wrapped in Sequential):
      - conv_t, conv_xy in the Stem (BN is in a different parent)
      - conv_b + norm_b  (norm_b is Sequential(BN, SE), not plain BN)
      - SE convolutions (norm_b.1.block.0/2 -- no BN after them)
      - post_conv in Head (no BN follows it)

    The un-folded BN layers remain as separate float operations at
    inference time on the SoC, matching the PTQ pipeline exactly.

    Returns the number of BN layers folded.
    """
    folded = 0

    for parent in model.modules():
        children = list(parent.named_children())
        for i in range(len(children) - 1):
            name_a, mod_a = children[i]
            name_b, mod_b = children[i + 1]
            if isinstance(mod_a, nn.Conv3d) and isinstance(mod_b, nn.BatchNorm3d):
                fold_bn_into_conv(mod_a, mod_b)
                setattr(parent, name_b, nn.Identity())
                folded += 1

    return folded


# ============================================================================
# Fake Quantization Modules (QAT building blocks)
# ============================================================================

class FakeQuantizePerTensor(nn.Module):
    """
    Fake quantization for activations: per-tensor, symmetric, int8.

    During forward pass:
        1. Track running min/max (EMA) to compute the scale
        2. Quantize: round(x / scale), clip to [-127, 127]
        3. Dequantize: q * scale
    This simulates quantization noise while keeping gradients flowing
    (straight-through estimator).
    """

    def __init__(self, ema_decay: float = 0.999):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer("running_max", torch.tensor(0.0))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

    @property
    def scale(self) -> float:
        rmax = float(self.running_max.item())
        if rmax == 0.0:
            return 1.0
        return rmax / 127.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_max = x.detach().abs().max()
            if self.num_batches == 0:
                self.running_max.copy_(batch_max)
            else:
                self.running_max.mul_(self.ema_decay).add_(
                    batch_max * (1 - self.ema_decay)
                )
            self.num_batches.add_(1)

        s = self.running_max / 127.0
        s = torch.clamp(s, min=1e-8)

        # Fake quantize with straight-through estimator (STE)
        x_q = torch.clamp(torch.round(x / s), -127, 127)
        x_dq = x_q * s
        return x + (x_dq - x).detach()


class FakeQuantizePerChannel(nn.Module):
    """
    Fake quantization for weights: per-output-channel, symmetric, int8.
    Applied to Conv3d weights with shape (O, I/g, kT, kH, kW).
    """

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        out_c = w.shape[0]
        flat = w.detach().reshape(out_c, -1)
        abs_max = flat.abs().max(dim=1).values
        abs_max = torch.clamp(abs_max, min=1e-8)
        scales = abs_max / 127.0
        bcast_shape = (out_c,) + (1,) * (w.dim() - 1)
        s = scales.reshape(bcast_shape)
        w_q = torch.clamp(torch.round(w / s), -127, 127)
        w_dq = w_q * s
        return w + (w_dq - w).detach()


# ============================================================================
# QAT Model Wrapping
# ============================================================================

class QATConv3dWrapper(nn.Module):
    """
    Wraps a Conv3d (already BN-folded) with fake quantization on input
    and weight. The original Conv3d is kept as a submodule so its
    parameters are visible to the optimizer.
    """

    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        self.conv = conv
        self.input_fq = FakeQuantizePerTensor()
        self.weight_fq = FakeQuantizePerChannel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_fq(x)
        w_fq = self.weight_fq(self.conv.weight)
        return nn.functional.conv3d(
            x, w_fq, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def _replace_conv3d_with_qat(parent: nn.Module, name: str, conv: nn.Conv3d) -> None:
    """Replace a Conv3d attribute with a QATConv3dWrapper."""
    setattr(parent, name, QATConv3dWrapper(conv))


def insert_fake_quant(model: nn.Module) -> int:
    """
    Walk the model and wrap every Conv3d with QATConv3dWrapper.
    BN must already be folded before calling this.

    Returns the number of Conv3d layers wrapped.
    """
    # Collect (parent, attr_name, conv_module) triples before modifying
    targets: List[Tuple[nn.Module, str, nn.Conv3d]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv3d):
                targets.append((parent, child_name, child))

    for parent, child_name, conv in targets:
        _replace_conv3d_with_qat(parent, child_name, conv)

    return len(targets)


def collect_qat_modules(model: nn.Module) -> Dict[str, QATConv3dWrapper]:
    """Return dict mapping full path -> QATConv3dWrapper for all wrapped layers."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, QATConv3dWrapper)
    }


# ============================================================================
# Export (matches PTQ format exactly)
# ============================================================================

def quantize_weight_per_channel(w: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8 quantization."""
    w_np = w.detach().cpu().float().numpy()
    out_c = w_np.shape[0]
    flat = w_np.reshape(out_c, -1)
    abs_max = np.max(np.abs(flat), axis=1)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = (abs_max / 127.0).astype(np.float32)
    bcast_shape = (out_c,) + (1,) * (w_np.ndim - 1)
    w_q = np.round(w_np / scales.reshape(bcast_shape))
    w_q = np.clip(w_q, -127, 127).astype(np.int8)
    return w_q, scales


def quantize_bias(
    b: torch.Tensor, input_scale: float, weight_scales: np.ndarray
) -> np.ndarray:
    """Quantize bias to int32 with scale = input_scale * weight_scale[c]."""
    b_np = b.detach().cpu().float().numpy()
    bias_scale = input_scale * weight_scales
    bias_scale = np.where(bias_scale == 0, 1.0, bias_scale)
    b_q = np.round(b_np / bias_scale)
    b_q = np.clip(b_q, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return b_q.astype(np.int32)


class _OutputObserver:
    """Simple hook-based absolute-max observer for calibration."""

    def __init__(self):
        self.abs_max = 0.0

    def __call__(self, _mod, _inp, output):
        if isinstance(output, torch.Tensor):
            val = float(output.detach().abs().max().item())
            if val > self.abs_max:
                self.abs_max = val

    @property
    def scale(self) -> float:
        if self.abs_max == 0.0:
            return 1.0
        return self.abs_max / 127.0


def export_qat_model(
    model: nn.Module,
    output_path: Path,
    calib_loader: DataLoader,
    num_calib_batches: int = 64,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Export a QAT-trained model to the same .npz format as the PTQ script.

    Steps:
    1. Collect all QATConv3dWrapper modules
    2. Run a calibration pass to measure output scales
    3. Quantize weights per-channel and biases to int32
    4. Save in the PTQ-compatible .npz format
    """
    model.eval()

    qat_modules = collect_qat_modules(model)
    out: Dict[str, np.ndarray] = {}
    n_layers = 0

    # --- Export the Head's final Linear (blocks.5.proj) as float32 ---
    # The PTQ script exports this with scratch-style keys: blocks.5.proj_weight/bias
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "proj" in name:
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            prefix = f"{parent_path}.proj" if parent_path else "proj"
            out[f"{prefix}_weight"] = (
                module.weight.detach().cpu().float().numpy()
            )
            if module.bias is not None:
                out[f"{prefix}_bias"] = (
                    module.bias.detach().cpu().float().numpy()
                )

    # --- Calibration pass for output scales ---
    print("  Running calibration for output scales ...")
    output_obs: Dict[str, _OutputObserver] = {}
    handles = []

    for name, qat_mod in qat_modules.items():
        obs = _OutputObserver()
        output_obs[name] = obs
        handles.append(qat_mod.register_forward_hook(obs))

    with torch.no_grad():
        for i, (batch, _) in enumerate(calib_loader):
            if i >= num_calib_batches:
                break
            batch = batch.to(device)
            _ = model(batch)
            if (i + 1) % 16 == 0:
                print(f"    calibrated {i+1}/{num_calib_batches}")

    for h in handles:
        h.remove()

    # --- Quantize and export each Conv3d ---
    for qat_name, qat_mod in qat_modules.items():
        conv = qat_mod.conv
        input_scale = np.float32(qat_mod.input_fq.scale)
        output_scale = np.float32(output_obs[qat_name].scale)

        w_q, w_scales = quantize_weight_per_channel(conv.weight)

        # The QATConv3dWrapper replaces the original Conv3d, so the QAT name
        # includes a trailing ".conv" for the inner module. The PTQ .npz uses
        # the ORIGINAL Conv3d path (without ".conv"). But since QATConv3dWrapper
        # is placed at the original Conv3d's position, the qat_name IS the
        # original path (e.g., "blocks.0.conv.conv_t" -> QATConv3dWrapper).
        # We need to strip nothing — the wrapper sits at the original name.
        layer_name = qat_name

        out[f"{layer_name}.weight_q"] = w_q
        out[f"{layer_name}.weight_scale"] = w_scales
        out[f"{layer_name}.input_scale"] = input_scale
        out[f"{layer_name}.output_scale"] = output_scale

        if conv.bias is not None:
            out[f"{layer_name}.bias_q"] = quantize_bias(
                conv.bias, float(input_scale), w_scales
            )

        n_layers += 1

    # --- Metadata ---
    out["__meta__.num_quantized_layers"] = np.int32(n_layers)
    out["__meta__.weight_scheme"] = np.array(
        "symmetric_per_channel_int8", dtype="S64"
    )
    out["__meta__.act_scheme"] = np.array(
        "symmetric_per_tensor_int8", dtype="S64"
    )
    out["__meta__.bias_scheme"] = np.array(
        "int32_input_scale_x_weight_scale", dtype="S64"
    )
    out["__meta__.training_method"] = np.array(
        "quantization_aware_training", dtype="S64"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    print(f"  Saved {n_layers} QAT-quantized layers to {output_path}")


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, top1_accuracy)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - start_time
            print(
                f"  [{epoch+1}/{num_epochs}] batch {batch_idx+1}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"acc={100.*correct/total:.1f}% "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"({elapsed:.1f}s)"
            )

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate. Returns (avg_loss, top1_accuracy, top5_accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        total += targets.size(0)

        _, pred1 = outputs.max(1)
        correct_1 += pred1.eq(targets).sum().item()

        _, pred5 = outputs.topk(min(5, outputs.size(1)), dim=1)
        correct_5 += sum(
            targets[i].item() in pred5[i].tolist()
            for i in range(targets.size(0))
        )

    avg_loss = running_loss / total if total > 0 else 0.0
    acc1 = 100.0 * correct_1 / total if total > 0 else 0.0
    acc5 = 100.0 * correct_5 / total if total > 0 else 0.0
    return avg_loss, acc1, acc5


# ============================================================================
# Model Loading
# ============================================================================

def load_model(input_path: Optional[Path] = None) -> nn.Module:
    """Load a float32 X3D-M either from the hub or a local checkpoint."""
    if input_path is None:
        print("Loading pretrained X3D-M from PyTorchVideo hub ...")
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
        )
    else:
        print(f"Loading X3D-M from {input_path} ...")
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_m", pretrained=False
        )
        try:
            sd = torch.load(input_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(input_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
    return model


def pick_device() -> torch.device:
    """Prefer MPS on Apple Silicon, then CUDA, then CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantization-Aware Training (QAT) for X3D-M.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Preparation:
  Organize Kinetics-400 (or a subset) as:
    data/kinetics400/train/<class_name>/<video_files>
    data/kinetics400/val/<class_name>/<video_files>

  Download from: https://github.com/cvdfoundation/kinetics-dataset

Example:
  python scripts/quantize_x3d_qat.py \\
      --train-dir data/kinetics400/train \\
      --val-dir data/kinetics400/val \\
      -o weights/x3d_m_int8_qat.npz
""",
    )

    # Data
    parser.add_argument(
        "--train-dir", type=Path, required=True,
        help="Path to training videos (class-organized subdirectories).",
    )
    parser.add_argument(
        "--val-dir", type=Path, default=None,
        help="Path to validation videos. If omitted, no validation is run.",
    )

    # Output
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("weights/x3d_m_int8_qat.npz"),
        help="Output .npz path for QAT-quantized weights.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints"),
        help="Directory for saving QAT training checkpoints.",
    )

    # Model
    parser.add_argument(
        "-i", "--input", type=Path, default=None,
        help="Optional input float32 .pth checkpoint. If omitted, load from hub.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume QAT from a checkpoint (.pth file).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of QAT fine-tuning epochs (default: 20).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4).")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Peak learning rate (default: 1e-5).")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4).")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="LR warmup epochs (default: 2).")

    # Data loading
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4).")
    parser.add_argument("--max-clips-per-class", type=int, default=0,
                        help="Max training clips per class (0 = all). "
                             "Useful for quick experiments.")
    parser.add_argument("--num-calib-batches", type=int, default=64,
                        help="Calibration batches for output scale estimation "
                             "(default: 64).")

    # Misc
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, mps, or cuda.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs (default: 5).")

    args = parser.parse_args()

    # --- Dependency checks ---
    if torch is None:
        print("Error: PyTorch is required. Install with: "
              "pip install torch torchvision", file=sys.stderr)
        return 1
    if cv2 is None:
        print("Error: OpenCV is required. Install with: "
              "pip install opencv-python", file=sys.stderr)
        return 1

    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")
    print(f"QAT config: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}")

    # --- Step 1: Load pretrained model ---
    model = load_model(args.input)
    model.eval()

    # --- Step 2: Fold ALL BatchNorm into Conv3d ---
    print("\nFolding BatchNorm into Conv3d (before QAT) ...")
    n_folded = fold_all_bn(model)
    print(f"  Folded {n_folded} BatchNorm layers")

    # --- Step 3: Insert fake quantization on every Conv3d ---
    print("Inserting fake-quantization nodes ...")
    n_wrapped = insert_fake_quant(model)
    print(f"  Wrapped {n_wrapped} Conv3d layers with fake quantization")

    # --- Build datasets ---
    print("\nBuilding training dataset ...")
    train_dataset = Kinetics400Dataset(
        args.train_dir,
        is_train=True,
        max_clips_per_class=args.max_clips_per_class,
    )

    val_dataset = None
    if args.val_dir is not None:
        print("Building validation dataset ...")
        val_dataset = Kinetics400Dataset(
            args.val_dir,
            is_train=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = model.to(device)

    # --- Optimizer and scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with linear warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}")

    # --- Step 4: QAT Training loop ---
    print(f"\n{'='*60}")
    print(f"Starting QAT training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    training_log = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs,
        )

        # Validate
        val_loss, val_acc1, val_acc5 = 0.0, 0.0, 0.0
        if val_loader is not None:
            val_loss, val_acc1, val_acc5 = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "time": epoch_time,
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)

        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.1f}%")
        if val_loader is not None:
            print(f"  Val:   loss={val_loss:.4f} "
                  f"top1={val_acc1:.1f}% top5={val_acc5:.1f}%")
        print()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = args.checkpoint_dir / f"qat_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc1": val_acc1,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Track best
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_path = args.checkpoint_dir / "qat_best.pth"
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc1": val_acc1,
            }, best_path)
            print(f"  New best! val_acc1={val_acc1:.1f}% -> {best_path}")

    # --- Step 5: Export QAT weights to .npz ---
    print(f"\n{'='*60}")
    print("Exporting QAT-trained model to int8 .npz ...")
    print(f"{'='*60}\n")

    export_qat_model(
        model,
        args.output,
        calib_loader=val_loader if val_loader is not None else train_loader,
        num_calib_batches=args.num_calib_batches,
        device=device,
    )

    # Save training log
    log_path = args.output.with_suffix(".json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")

    print(f"\nDone! QAT weights saved to {args.output}")
    if best_val_acc > 0:
        print(f"Best validation accuracy: {best_val_acc:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
