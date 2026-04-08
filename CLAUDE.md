# X3D-M: PyTorch-Free Video Classification for RISC-V SoC

## Project Overview

X3D-M (eXpand-3D Medium) video classification model implemented from scratch using **NumPy and OpenCV** — no PyTorch at runtime. Designed to run on the **Microchip PolarFire SoC Icicle Kit** (RISC-V), where PyTorch is unavailable.

- **Input:** `(B, 3, 16, 224, 224)` — batch, RGB, 16 frames, 224×224 spatial
- **Output:** `(B, 400)` — Kinetics-400 action recognition class logits
- **~3.79M parameters**
- **Target hardware:** PolarFire SoC with 4× U54 RISC-V cores (RV64GC, no vector extension), 2GB LPDDR4, 32KB L1 cache per core, 2MB shared L2

## Architecture

### Two-Layer Design: `ops/` vs `nn/`

The `scratch/` package mirrors PyTorch's design:

- **`scratch/ops/`** — Stateless pure functions (NumPy arrays in/out). Mathematical building blocks: `conv3d_forward()`, `batchnorm3d_forward()`, `relu()`, `silu()`, `sigmoid()`, `avg_pool3d_forward()`, `linear_forward()`, `dropout_forward()`.
- **`scratch/nn/`** — Stateful module classes wrapping ops with learnable parameters. Inherit from `Module` base class. Store params in `_parameters` dict, children in `_modules` dict.

### Key Module Hierarchy

```
X3D_M (scratch/models/x3d_m.py)
  └── blocks: ModuleList[0..5]
        [0] Stem (nn/stem.py) — (2+1)D factorized: conv_t(1×3×3) + conv_xy(5×1×1 depthwise) + BN + ReLU
        [1] ResStage(depth=3,  in=24,  inner=54,  out=24)   — Stage 2
        [2] ResStage(depth=5,  in=24,  inner=108, out=48)   — Stage 3
        [3] ResStage(depth=11, in=48,  inner=216, out=96)   — Stage 4
        [4] ResStage(depth=7,  in=96,  inner=432, out=192)  — Stage 5
        [5] Head (nn/head.py) — ProjectedPool(192→432→2048) + Dropout + Linear(2048→400)
```

Each ResStage contains ResBlock instances. Each ResBlock has:
- **branch2** (BottleneckBlock): conv_a(1×1×1 pointwise) → BN → ReLU → conv_b(3×3×3 depthwise) → BN+SE → SiLU → conv_c(1×1×1 pointwise) → BN
- **branch1** (skip connection): Identity or 1×1×1 Conv3d (when dimensions change)
- Output: `ReLU(branch2(x) + branch1(x))`

SE (Squeeze-and-Excitation) is applied on even-indexed blocks (0, 2, 4, ...). SE ratio = 0.0625 (1/16), bottleneck width rounded to nearest multiple of 8.

### Convolution Methods

Four implementations in `scratch/ops/conv3d.py`, selectable via `--method` or `set_conv3d_method()`:

| Method | Description | When to use |
|--------|-------------|-------------|
| `slow` | Pure NumPy, 6-deep nested loops | Correctness reference only |
| `fast` | Multi-threaded OpenCV `cv2.filter2D` **(default)** | General use |
| `threaded` | Same as fast (legacy alias) | Same as fast |
| `native` | C shared library via ctypes (pthreads, spatial tiling) | Production on SoC |

The `fast` method uses adaptive hybrid parallelism:
- Pointwise/standard convolutions → output-channel parallelism (`_conv3d_oc_parallel`)
- Depthwise convolutions → temporal parallelism (`_conv3d_temporal_parallel`)

### C Backend (`scratch/ops/conv3d_c/`)

Targets RV64GC with `-O3 -funroll-loops -ffast-math`. Uses pthreads (4 threads for 4 U54 cores). Spatial tiling `TILE_H=8, TILE_W=16` fits in 32KB L1. Three fast paths: pointwise, depthwise, general.

Build: `make -C scratch/ops/conv3d_c` (auto-detects arch), `make -C scratch/ops/conv3d_c riscv`, or `make -C scratch/ops/conv3d_c native`.

## File Layout

```
x3d/
├── main.py                  # Inference entry point with profiling (CLI)
├── visualize_stats.py       # Cross-platform profiling comparison & charts
├── x3d_layers.py            # PyTorch reference implementation (verification only)
├── scratch/                 # The PyTorch-free library
│   ├── __init__.py          # Public API: X3D_M, set_conv3d_method, etc.
│   ├── ops/                 # Stateless math operations
│   │   ├── conv3d.py        # 4 conv methods + threading helpers
│   │   ├── conv3d_c/        # C backend (conv3d.c, conv3d.h, Makefile)
│   │   ├── batchnorm3d.py
│   │   ├── activations.py   # relu, silu, sigmoid
│   │   ├── pooling.py       # avg_pool3d, adaptive_avg_pool3d
│   │   ├── linear.py
│   │   └── dropout.py
│   ├── nn/                  # Stateful modules (Module subclasses)
│   │   ├── module.py        # Base class (like torch.nn.Module)
│   │   ├── sequential.py    # Sequential + ModuleList
│   │   ├── conv3d.py        # Conv3d layer
│   │   ├── batchnorm3d.py   # BatchNorm3d layer
│   │   ├── squeeze_excitation.py
│   │   ├── bottleneck.py    # conv_a → conv_b → conv_c pipeline
│   │   ├── resblock.py      # Bottleneck + skip connection
│   │   ├── resstage.py      # Stage of ResBlocks
│   │   ├── stem.py          # (2+1)D factorized stem
│   │   └── head.py          # Classification head
│   ├── models/x3d_m.py      # Full model assembly
│   ├── load_weights.py      # Load .npz weights (key remapping)
│   └── stats.py             # StatsCollector, FLOPs estimation
├── scripts/
│   └── convert_pytorch_weights_to_numpy.py  # PyTorch → .npz (run on dev machine)
├── weights/                 # .npz files (gitignored, large)
├── run_stats/               # Profiling JSON/TXT output (gitignored)
├── archive/                 # Legacy C++/PyTorch code (not used)
└── DOCUMENTATION.md         # Comprehensive technical documentation
```

## Common Commands

```bash
# Basic inference
python main.py
python main.py --stem-only

# Profiled inference
python main.py --profile
python main.py --profile --notes "PolarFire SoC Icicle Kit"
python main.py --profile --stem-only
python main.py --profile --output-dir my_stats

# Select convolution method
python main.py --method fast              # default (multi-threaded OpenCV)
python main.py --method native --profile  # C backend

# Build C backend
make -C scratch/ops/conv3d_c              # auto-detect
make -C scratch/ops/conv3d_c riscv        # RISC-V target
make -C scratch/ops/conv3d_c native       # x86 native
make -C scratch/ops/conv3d_c clean

# Weight conversion (requires PyTorch on dev machine)
python scripts/convert_pytorch_weights_to_numpy.py -o weights/x3d_m_kinetics400.npz

# Visualization
python visualize_stats.py
python visualize_stats.py --html
python visualize_stats.py --output charts --format png
python visualize_stats.py --no-charts --top-n 15
```

## Conventions

- **Tensor layout:** Always `(B, C, T, H, W)` — batch, channels, time, height, width
- **Data type:** `float32` throughout
- **Temporal dimension preserved:** T=16 is never downsampled; only spatial dims (H, W) are reduced by stride-2
- **Weight key names** must match PyTorchVideo's `blocks.N.…` naming for `load_pretrained_numpy` to work. See key remapping in `scripts/convert_pytorch_weights_to_numpy.py`
- **Stem naming quirk:** `conv_t` is the spatial conv (1×3×3), `conv_xy` is the temporal conv (5×1×1) — inherited from PyTorchVideo
- **SE on even blocks only:** blocks 0, 2, 4, ... get SqueezeExcitation; odd blocks get Identity
- **No bias before BatchNorm:** Conv3d layers followed by BN use `bias=False` (BN absorbs it)
- **Thread count:** `NUM_THREADS = 4` (hardcoded for PolarFire SoC's 4 U54 cores)
- **Module.eval()** must be called before inference (affects BN running stats and dropout)
- **Dependencies:** NumPy, OpenCV (opencv-python). Optional: matplotlib (for visualize_stats.py). GCC for C backend.

## Testing & Verification

- Compare scratch output with PyTorch reference: run `python x3d_layers.py` (requires PyTorch)
- Outputs should match within floating-point tolerance ~1e-4
- Profile runs save to `run_stats/` as JSON + human-readable TXT

## Key Optimization Notes

- Convolution accounts for 80-90% of total inference time
- Multi-threading gives ~2.5-3.5x speedup on convolutions (4 cores)
- Possible future optimizations: BN fusion into conv weights, im2col+GEMM for slow path, pre-allocated padding buffers, int8 quantization
- Memory bandwidth is the primary bottleneck on PolarFire SoC (32-bit LPDDR4)
- Stage 4 (11 blocks, 216 inner channels) and Stage 5 (7 blocks, 432 inner channels) are the most compute-heavy
