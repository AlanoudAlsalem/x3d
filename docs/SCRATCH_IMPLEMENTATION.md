# Scratch 3D CNN Implementation (No PyTorch)

Minimalistic reimplementation of X3D-M layers and model using **NumPy and OpenCV**, for deployment on systems where PyTorch is not available (e.g. Microchip PolarFire Icicle Kit RISC-V SoC running Ubuntu).

## Overview

- **Purpose:** Run X3D-M–style 3D CNNs without PyTorch; reuse primitives for other 3D CNN models.
- **Data layout:** All 3D tensors use `(B, C, T, H, W)` (batch, channels, time, height, width).
- **Dependencies:** NumPy, OpenCV (opencv-python). No PyTorch, no scipy. A pure-NumPy fallback (`conv3d_forward_slow`) is available for platforms without OpenCV.

## Directory Layout

```
scratch/
├── __init__.py          # Package root; exports Module, X3D_M
├── ops/                 # Low-level operations (no learned parameters)
│   ├── __init__.py
│   ├── conv3d.py        # 3D convolution (standard + depthwise)
│   ├── batchnorm3d.py   # 3D batch normalization
│   ├── activations.py   # ReLU, SiLU, Sigmoid
│   ├── pooling.py      # AvgPool3d, AdaptiveAvgPool3d
│   ├── linear.py       # Fully connected (linear) layer
│   └── dropout.py      # Dropout
├── nn/                  # Layers (with parameters) and building blocks
│   ├── __init__.py
│   ├── module.py       # Base class Module (train/eval, parameters)
│   ├── sequential.py   # Sequential, ModuleList
│   ├── conv3d.py       # Conv3d layer (wraps ops + weight/bias)
│   ├── batchnorm3d.py   # BatchNorm3d layer
│   ├── squeeze_excitation.py  # SE block
│   ├── bottleneck.py   # BottleneckBlock, Identity
│   ├── resblock.py     # ResBlock (bottleneck + skip)
│   ├── resstage.py     # ResStage (sequence of ResBlocks)
│   ├── stem.py         # Conv2plus1dStem, Stem
│   └── head.py         # ProjectedPool, Head
├── load_weights.py     # Load .npz weights (NumPy only; for SoC)
└── models/
    ├── __init__.py
    └── x3d_m.py        # Full X3D_M model
```

**scripts/** (run on laptop with PyTorch):

- **`scripts/convert_pytorch_weights_to_numpy.py`** – Reads PyTorch state_dict (hub or .pth), converts to NumPy, saves .npz for the scratch model.

Entry point: **`x3d_layers_scratch.py`** at project root (builds X3D-M and runs forward).

## Usage

### Build and run X3D-M

```python
from x3d_layers_scratch import build_x3d_m, run_forward
import numpy as np

model = build_x3d_m(num_classes=400)
model.eval()
x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
logits = run_forward(model, x)  # shape (1, 400)
```

### Use individual layers (e.g. for other 3D CNNs)

```python
from scratch.nn import Conv3d, BatchNorm3d, ResBlock, ResStage, Stem, Head
from scratch.ops import relu, silu, sigmoid, avg_pool3d_forward
```

## API Reference (docstrings summary)

### Base

- **`Module`** – Base class. Implements `forward(x)`, `parameters()`, `train(mode)`, `eval()`.

### Ops (no parameters)

- **`conv3d_forward(x, weight, bias, stride, padding, groups)`** – 3D conv; `bias` can be `None`; `groups=in_channels` for depthwise.
- **`batchnorm3d_forward(x, weight, bias, running_mean, running_var, eps, training, momentum)`** – BN over (B,T,H,W) per channel.
- **`relu(x)`**, **`silu(x)`**, **`sigmoid(x)`** – Element-wise activations.
- **`avg_pool3d_forward(x, kernel_size, stride=None)`** – 3D average pooling.
- **`adaptive_avg_pool3d_forward(x, output_size)`** – Adaptive avg pool; `output_size=1` for global pool.
- **`linear_forward(x, weight, bias)`** – Dense: `x @ weight.T + bias`.
- **`dropout_forward(x, p, training, rng=None)`** – Dropout (no-op when `training=False`).

### NN layers (with parameters)

- **`Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1)`** – 3D conv layer; `forward(x)`.
- **`BatchNorm3d(num_features, eps=1e-5, momentum=0.1)`** – 3D BN; uses running mean/var in eval.
- **`Sequential(*modules)`**, **`ModuleList(modules)`** – Sequential chain and list of modules.
- **`SqueezeExcitation(channels, se_ratio=0.0625)`** – SE block (global pool → 1×1×1 convs → scale).
- **`BottleneckBlock(in_channels, inner_channels, out_channels, stride=1, use_se=False, se_ratio=0.0625)`** – X3D bottleneck (1×1→3×3 DW→1×1 + BN + ReLU/SiLU).
- **`ResBlock(in_channels, inner_channels, out_channels, stride=1, use_se=False, se_ratio=0.0625)`** – Residual block (bottleneck + shortcut).
- **`ResStage(depth, in_channels, inner_channels, out_channels, stride=2, se_ratio=0.0625)`** – Stack of ResBlocks; first block uses `stride`, SE on even indices.
- **`Conv2plus1dStem()`** – (2+1)D stem (spatial then temporal conv).
- **`Stem()`** – Stem + BN + ReLU.
- **`ProjectedPool()`** – pre_conv → pool → post_conv (192→432→2048).
- **`Head(num_classes=400)`** – ProjectedPool + dropout + linear → logits.
- **`X3D_M(num_classes=400)`** – Full model: Stem → 4 ResStages → Head; `forward(x)` returns `(B, num_classes)`.

### Entry and weights

- **`build_x3d_m(num_classes=400, weights_path=None, strict_weights=True)`** – Returns an `X3D_M` instance; if `weights_path` is set, loads pretrained .npz (for SoC).
- **`run_forward(model, x)`** – Runs `model.forward(x)`; `x` shape `(B, 3, 16, 224, 224)`.
- **`load_pretrained_numpy(model, path_or_archive, strict=..., verbose=...)`** – Load from .npz or dict; returns (missing_keys, unexpected_keys). NumPy only.
- **`load_pretrained_numpy_if_available(model, path, verbose=...)`** – Load if file exists; no-op otherwise. Use on SoC when weights may or may not be present.

## Tensor shapes (X3D-M)

| Stage        | Shape (B=1)        |
|-------------|--------------------|
| Input       | (1, 3, 16, 224, 224) |
| After Stem  | (1, 24, 16, 112, 112) |
| After Stage 2 | (1, 24, 16, 56, 56) |
| After Stage 3 | (1, 48, 16, 28, 28) |
| After Stage 4 | (1, 96, 16, 14, 14) |
| After Stage 5 | (1, 192, 16, 7, 7) |
| Output      | (1, 400)            |

## Pretrained weights (conversion on laptop, loading on SoC)

PyTorch cannot run on the PolarFire RISC-V SoC, so pretrained weights are handled in two steps.

### 1. Convert on a machine with PyTorch (laptop)

Run the conversion script to download the pretrained X3D-M (or load a local .pth) and save a single .npz file:

```bash
# From project root; requires: pip install torch
python scripts/convert_pytorch_weights_to_numpy.py -o weights/x3d_m_kinetics400.npz
```

This uses `torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)` and writes NumPy arrays with keys that match the scratch module hierarchy (e.g. `blocks.0.conv.conv_t.weight`, `blocks.5.proj_weight`). To use a local checkpoint instead:

```bash
python scripts/convert_pytorch_weights_to_numpy.py -i path/to/x3d_m.pth -o weights/x3d_m.npz
```

Copy the resulting **.npz file** to the SoC (e.g. via USB, NFS, or SCP).

### 2. Load on the SoC (NumPy only)

On the SoC, load the copied .npz with no PyTorch dependency:

```python
from x3d_layers_scratch import build_x3d_m, run_forward
# Option A: load at build time
model = build_x3d_m(num_classes=400, weights_path="weights/x3d_m_kinetics400.npz")
model.eval()
# Option B: load only if file exists (e.g. in a script that also runs without weights)
from scratch.load_weights import load_pretrained_numpy_if_available
model = build_x3d_m(num_classes=400)
load_pretrained_numpy_if_available(model, "weights/x3d_m_kinetics400.npz")
model.eval()
```

The loader in `scratch/load_weights.py` uses only NumPy: it opens the .npz, walks the module tree by key path, and assigns arrays into each module’s `_parameters`. Key names in the .npz are rewritten from the PyTorch state_dict (e.g. `blocks.5.proj.weight` → `blocks.5.proj_weight`, and SE `norm_b.1.block.0` / `block.2` → `norm_b.1.conv1` / `conv2`) so they match the scratch layout.

## Reusing for other 3D CNNs

- **Ops** in `scratch/ops/` are generic (conv3d, batchnorm3d, activations, pooling, linear, dropout). Use them to implement other backbones.
- **Building blocks** in `scratch/nn/` (Conv3d, BatchNorm3d, ResBlock, ResStage, Stem, Head) are modular; you can compose new models in `scratch/models/` (e.g. different stage widths/depths or different stems/heads) and keep using the same ops and base `Module` class.

## Notes

- **Inference:** Call `model.eval()` so BatchNorm uses running statistics and Dropout is disabled.
- **Performance:** The current conv3d and pooling implementations use simple loops and are aimed at correctness and portability; they can be slow on large inputs (e.g. one stem forward at 224×224 may take tens of seconds). For production on RISC-V, replace the core loops in `scratch/ops/conv3d.py` and `scratch/ops/pooling.py` with optimized C/assembly or FPGA-offloaded kernels while keeping the same Python API and `scratch` layout.
- **Numerics:** Same formulas as the PyTorch reference (BN eps=1e-5, momentum=0.1; same strides/padding). For exact parity you would need to load the same weights and use the same floating-point behavior.
