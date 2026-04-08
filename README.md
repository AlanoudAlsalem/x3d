# X3D-M: PyTorch-Free Implementation for RISC-V SoC

X3D-M (eXpand-3D Medium) video classification model built from scratch using **NumPy and OpenCV**—no PyTorch. Designed for platforms where PyTorch is unavailable, such as the PolarFire SoC Icicle Kit (RISC-V).

- **Input:** Video clip `(B, 3, 16, 224, 224)` — batch, RGB, 16 frames, 224×224 spatial
- **Output:** Class logits `(B, 400)` for Kinetics-400 action recognition
- **~3.79M parameters**

---

## Requirements

- Python 3.8+
- NumPy
- OpenCV (opencv-python) — used for accelerated 3D convolution via `cv2.filter2D`
- GCC (for building the optional C backend — `scratch/ops/conv3d_c/`)

Optional (for visualization):
- matplotlib (for charts)

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd x3d

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
x3d/
├── main.py              # Inference entry point with profiling
├── visualize_stats.py   # Statistics visualization and comparison
├── x3d_layers.py        # PyTorch reference implementation
├── scratch/             # PyTorch-free implementation
│   ├── models/         # X3D-M model definition
│   ├── nn/             # Neural network layers (Conv3d, BatchNorm, etc.)
│   ├── ops/            # Low-level operations (conv3d, pooling, etc.)
│   │   ├── conv3d_c/   # Float32 C backend (pthreads, tiling)
│   │   ├── conv3d_fpga_c/  # Int8 FPGA-offload C backend
│   │   │   ├── conv3d_fpga.c
│   │   │   ├── conv3d_fpga.h
│   │   │   └── Makefile
│   │   └── conv3d_fpga.py  # Python ctypes wrapper for FPGA backend
│   ├── load_weights.py # Load .npz weights into model
│   └── stats.py        # Profiling and statistics collection
├── fpga_tests/          # Per-layer FPGA offload validation
│   ├── test_layer.py   # CLI: runs float, SW-int8, Sim, and HW paths
│   ├── kernels.py      # sw_int8, fpga_sim, fpga_hw conv implementations
│   ├── quant.py        # Int8 quantization / requantization primitives
│   └── layer_configs.py # Conv layer specs for all X3D-M conv types
├── scripts/
│   └── convert_pytorch_weights_to_numpy.py  # Convert PyTorch weights to .npz
├── weights/            # Model weights (.npz) — not in repo
├── run_stats/          # Generated profiling statistics — not in repo
└── docs/
```

---

## Scratch Library

The `scratch` package provides a pure NumPy implementation of X3D-M.

### Building the Model

```python
from main import build_x3d_m, run_forward
import numpy as np

# Build with random weights
model = build_x3d_m(num_classes=400)
model.eval()

# Build with pretrained weights (from .npz file)
model = build_x3d_m(
    num_classes=400,
    weights_path="weights/x3d_m_kinetics400.npz"
)
model.eval()

# Run inference
x = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
logits = run_forward(model, x)
```

### Using Scratch Modules Directly

```python
from scratch.models.x3d_m import X3D_M
from scratch.load_weights import load_pretrained_numpy

model = X3D_M(num_classes=400)
load_pretrained_numpy(model, "weights/x3d_m_kinetics400.npz")
model.eval()
```

---

## Inference

### Basic Inference (No Profiling)

```bash
# Run stem + full forward pass (slow with reference NumPy conv)
python main.py

# Run stem only (quick check)
python main.py --stem-only
```

### Inference with Profiling and Logging

```bash
# Full model profiling — logs every layer to terminal and saves to run_stats/
python main.py --profile

# Add notes for the run
python main.py --profile --notes "PolarFire SoC Icicle Kit"

# Stem only (quick profiling test)
python main.py --profile --stem-only

# Custom output directory
python main.py --profile --output-dir my_stats
```

**Output:**
- Terminal: Per-layer latency, params, FLOPs, output shapes
- `run_stats/<run_id>_<platform>.json` — machine-readable statistics
- `run_stats/<run_id>_<platform>.txt` — human-readable report

**Platform detection:** Automatically identifies macOS, PolarFire SoC, RISC-V Linux, etc.

---

## Weight Conversion

Weights must be converted from PyTorch to NumPy on a machine with PyTorch, then copied to the target device.

```bash
mkdir -p weights

# From PyTorchVideo hub (requires network)
python scripts/convert_pytorch_weights_to_numpy.py -o weights/x3d_m_kinetics400.npz

# From local checkpoint
python scripts/convert_pytorch_weights_to_numpy.py -i path/to/x3d_m.pth -o weights/x3d_m.npz
```

Copy the resulting `.npz` file to `weights/` on your target.

---

## Visualization Script

Analyze and compare profiling runs across platforms.

```bash
# Analyze all runs in run_stats/
python visualize_stats.py

# Custom stats directory
python visualize_stats.py --dir my_stats

# Save charts to directory
python visualize_stats.py --output charts --format png

# Generate HTML report
python visualize_stats.py --html

# Text output only (no matplotlib)
python visualize_stats.py --no-charts

# Show top N bottleneck layers
python visualize_stats.py --top-n 15
```

**Output:**
- Platform comparison tables
- Section breakdown by latency
- Bottleneck analysis (slowest layers)
- Charts: latency comparison, section breakdown, pie charts, speedup
- Optional HTML report

---

## Workflow

1. **On Mac/Linux (with PyTorch):** Convert weights and run profiling locally.
2. **On PolarFire SoC:** Copy `weights/` and run `main.py --profile`.
3. **Compare:** Copy `run_stats/*.json` from SoC back to your machine, run `visualize_stats.py` to compare.

---

## Convolution Methods

Four implementations are available, selectable via `--method` flag or `set_conv3d_method()`:

| Method | Description |
|--------|-------------|
| `slow` | Pure NumPy fallback (no OpenCV needed) |
| `fast` | Single-threaded OpenCV `cv2.filter2D` **(default)** |
| `threaded` | Multi-threaded OpenCV (4 Python threads, adaptive parallelism) |
| `native` | C shared library via ctypes (pthreads, spatial tiling, fast paths) |

### CLI usage

```bash
python main.py --method fast              # default
python main.py --method threaded          # Python multi-threaded
python main.py --method native --profile  # C backend
```

### Python usage

```python
from scratch import set_conv3d_method
set_conv3d_method("native")   # use C backend for all subsequent calls
```

### Building the C backend (native)

```bash
# Auto-detect architecture (RISC-V, x86_64, or ARM)
make -C scratch/ops/conv3d_c

# Explicit RISC-V target (on the PolarFire SoC or cross-compile)
make -C scratch/ops/conv3d_c riscv

# Explicit x86 native target (for testing on dev machine)
make -C scratch/ops/conv3d_c native
```

The C backend uses pthreads (4 threads for the PolarFire SoC's 4 U54 cores), cache-friendly spatial tiling for the 32 KiB L1, and separate fast paths for pointwise (1×1×1), depthwise, and general convolutions. It targets RV64GC (no vector extension) with `-O3 -funroll-loops -ffast-math`.

If `libconv3d.so` is not compiled, a message is printed at import time and `set_conv3d_method("native")` will raise a clear error.

---

## FPGA Offload (Int8 Convolution)

The FPGA offload path moves 3D convolutions off the RISC-V CPU and onto the
PolarFire FPGA fabric using **int8 arithmetic** — the same datapath the real
hardware accelerator will use.

### Architecture

The int8 pipeline has three implementations with a **unified API** so they can be compared directly:

| Implementation | Where it runs | Arithmetic | Requantization | Purpose |
|---|---|---|---|---|
| `sw_int8_conv3d` | CPU (Python) | float32→int32 | float `M[c]` | Gold-standard reference |
| `fpga_sim_int8_conv3d` | CPU (Python) | float32→int32 | fixed-point `(M0, n)` | Python FPGA simulator |
| `fpga_hw_int8_conv3d` | CPU (C lib) | **int8→int32** | fixed-point `(M0, n)` | **FPGA offload backend** |

All three accept the same signature: `(x_q, W_q, <requant_params>, stride, padding, groups) -> int8`.

The C backend (`scratch/ops/conv3d_fpga_c/`) mirrors the float32 C backend's
structure — pthreads (4 threads), spatial tiling, pointwise/depthwise/general
fast paths — but operates entirely in int8/int32.

### Building the FPGA C backend

```bash
make -C scratch/ops/conv3d_fpga_c            # auto-detect architecture
make -C scratch/ops/conv3d_fpga_c riscv      # RISC-V target
make -C scratch/ops/conv3d_fpga_c native     # x86 target (dev machine testing)
```

### Per-layer validation

```bash
# Run all four paths (float, SW-int8, FPGA-sim, FPGA-HW) for conv_b
python -m fpga_tests.test_layer

# Test other conv types
python -m fpga_tests.test_layer --layer conv_a
python -m fpga_tests.test_layer --layer conv_t --seed 7

# Skip FPGA HW if C backend not compiled
python -m fpga_tests.test_layer --skip-fpga-hw
```

The test validates that:
- **Sim vs SW** int8 disagreement ≤ 2 LSB (float-vs-fixed-point rounding)
- **HW vs Sim** produces **bit-identical** output (same requantization logic)

See `fpga_tests/README.md` for full details.

---

## License

See repository for license details.
