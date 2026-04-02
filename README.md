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
│   │   └── conv3d_c/   # C shared-library backend (pthreads, tiling)
│   │       ├── conv3d.c
│   │       ├── conv3d.h
│   │       └── Makefile
│   ├── load_weights.py # Load .npz weights into model
│   └── stats.py        # Profiling and statistics collection
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

## License

See repository for license details.
