# FPGA offload bring-up: per-layer int8 conv tests

This folder validates the int8 FPGA-offload path one convolution at a time.
It implements §7 ("tensor-by-tensor dataflow") and §10.4 ("dump int8 tensors
at boundaries") of `../fpga_flow.md`.

The first target is `conv_b` (the 3×3×3 depthwise inside every bottleneck),
but the harness is fully generic — `conv_a`, `conv_c`, `conv_t`, and
`conv_xy` are all supported out of the box.

---

## 1. What problem this solves

On the PolarFire SoC we want to move 3D convolutions off the RISC-V CPU and
onto the FPGA fabric. The FPGA will only ever see **int8** data. We need to
answer three questions:

1. **Does our quantization math work?** Given a float32 input and a
   float32-trained conv, can we compute an int8 input, int8 weights, run an
   int8 conv, and get back a float32 output that is close to the original
   float32 result?
2. **Does the FPGA's fixed-point requantizer agree with the "obvious"
   float32 implementation?** The real hardware will use `(M0, n)` shift-and-
   multiply instead of float multiply.
3. **Does the C backend (FPGA offload library) agree bit-for-bit with the
   Python FPGA simulator?** The C library mirrors the exact same int8×int8
   accumulation and (M0, n) requantization that the FPGA performs, but runs
   on the CPU. When the real FPGA fabric is wired up, this C call gets
   replaced by a DMA transfer — everything else stays identical.

This folder answers all three. It runs the exact same conv layer four ways:

| path                      | where it runs | arithmetic   | requantize  | purpose                        |
|---------------------------|--------------|--------------|-------------|--------------------------------|
| float reference           | CPU          | float32      | none        | accuracy ceiling              |
| software int8 (gold)      | CPU          | float32→int32| float `M`   | "correct" int8 output          |
| FPGA simulator            | CPU (Python) | float32→int32| fixed-pt    | Python stand-in for the FPGA   |
| **FPGA HW (C backend)**   | CPU (C lib)  | **int8→int32**| **fixed-pt**| **FPGA offload library**       |
| *(later)* real FPGA       | FPGA fabric  | int8→int32   | fixed-pt    | must match the C backend bit-for-bit |

---

## 2. How to run it

### Prerequisites

Build the FPGA int8 C backend:

```bash
make -C scratch/ops/conv3d_fpga_c        # auto-detect architecture
make -C scratch/ops/conv3d_fpga_c riscv   # explicit RISC-V target
make -C scratch/ops/conv3d_fpga_c native  # explicit x86 target
```

### Running tests

```bash
# defaults: --layer conv_b --seed 42 --out-dir fpga_tests/runs --tol-lsb 2
python -m fpga_tests.test_layer

# test a different conv type
python -m fpga_tests.test_layer --layer conv_a
python -m fpga_tests.test_layer --layer conv_t --seed 7

# skip FPGA HW path (if C backend not compiled)
python -m fpga_tests.test_layer --skip-fpga-hw
```

Each run prints a one-screen summary and writes two files into
`fpga_tests/runs/`:

* `<layer>_seed<N>.npz` — every tensor, scale, and multiplier from the run.
* `<layer>_seed<N>.json` — a human-readable summary with all the diff
  statistics and pass/fail flags.

Re-running with the same seed produces bit-identical results.

### Pass criteria

- **Sim vs SW:** The Python FPGA simulator and the software reference must
  disagree by at most `--tol-lsb` int8 levels (default 2). This absorbs
  float-vs-fixed-point rounding differences.
- **HW vs Sim:** The C backend must produce **bit-identical** output to the
  Python FPGA simulator (0 LSB tolerance). Both implement the exact same
  fixed-point (M0, n) requantization.

---

## 3. File layout

```
fpga_tests/
├── README.md            # this file
├── __init__.py
├── layer_configs.py     # one LayerConfig entry per conv type
├── quant.py             # symmetric int8 quantize / dequantize / scales,
│                        # float-M and fixed-point (M0, n) requantization
├── kernels.py           # sw_int8_conv3d, fpga_sim_int8_conv3d, and
│                        # fpga_hw_int8_conv3d (C backend via ctypes)
├── test_layer.py        # CLI entry point. Builds layer, runs all four
│                        # paths, saves everything, prints diff stats
└── runs/                # output tensors + JSON summaries (gitignored)
```

C backend (FPGA offload library):

```
scratch/ops/conv3d_fpga_c/
├── conv3d_fpga.h        # public C API
├── conv3d_fpga.c        # int8×int8→int32 conv + (M0,n) requantize, pthreads
└── Makefile             # auto-detect / riscv / native targets
```

Python wrapper:

```
scratch/ops/conv3d_fpga.py   # ctypes bindings for libconv3d_fpga.so
```

---

## 4. What a single run does, step by step

This maps 1:1 onto §7 of `../fpga_flow.md`. For a layer `L`:

1. **Build a float `Conv3d`** from the `LayerConfig` in `layer_configs.py`.
   Weights are Xavier-initialised from a fixed NumPy seed so every run is
   reproducible.
2. **Generate a reproducible float input** `x_f32` with
   `np.random.default_rng(seed).standard_normal(shape)`.
3. **Run the float conv once** → `y_ref_f32`. This is "Path A" from the
   doc, and also the thing we look at to choose `s_out`.
4. **Compute quantization parameters**:
   * `s_in  = max(|x_f32|) / 127`            — per-tensor
   * `s_w[c] = max(|W[c]|) / 127` for each output channel `c` — per-channel
   * `s_out = max(|y_ref_f32|) / 127`        — per-tensor
   * `M[c] = (s_in * s_w[c]) / s_out`        — float32 multiplier
   * `(M0[c], n[c])`                         — fixed-point form of `M[c]`
5. **Quantize** `x_f32 → x_q` (int8) and `W → W_q` (int8 per-channel).
6. **Run three int8 convs**:
   * *Software gold*: integer accumulate → `round(acc32 * M[c])` in float32
     → clip → int8. This is `sw_int8_conv3d`.
   * *FPGA sim*: integer accumulate → `(acc32 * M0[c]) >> n[c]` with
     rounding → clip → int8. This is `fpga_sim_int8_conv3d`.
   * *FPGA HW*: native int8×int8 → int32 accumulate → same (M0, n)
     requantize → int8. This is `fpga_hw_int8_conv3d` (C backend).
7. **Dequantize** all int8 outputs by multiplying by `s_out`.
8. **Compare** diffs and save all tensors to an `.npz`:
   * Sim vs SW (int8 space) — should be ≤ tolerance
   * HW vs Sim (int8 space) — should be exactly 0
   * SW int8 vs float reference — quantization-error budget
   * HW int8 vs float reference — total error budget

### The integer-accumulate trick (SW and Sim paths)

The SW and Sim paths reuse `scratch.ops.conv3d.conv3d_forward`, passing int8
values as float32. The float32 kernel produces exact integer accumulators as
long as the intermediate sums fit in the 24-bit float mantissa — which they
do for every layer in X3D-M. The C backend (HW path) does true int8×int8→int32
arithmetic natively.

---

## 5. Adding a new conv type

Adding, say, a hypothetical `conv_d` is literally one dict entry in
`layer_configs.py`:

```python
"conv_d": LayerConfig(
    name="conv_d",
    in_channels=96,
    out_channels=96,
    kernel_size=(3, 3, 3),
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    groups=96,
    input_shape=(1, 96, 16, 14, 14),
    description="Stage 4 depthwise experiment.",
),
```

Then `python -m fpga_tests.test_layer --layer conv_d` just works. No other
file needs to change.

---

## 6. Swapping in the real FPGA later

When the real FPGA int8 accelerator fabric is wired up:

1. Replace the body of `fpga_hw_int8_conv3d` in `scratch/ops/conv3d_fpga.py`
   with a DMA wrapper that takes `(x_q, W_q, M0, n, stride, padding,
   groups)`, DMAs into FPGA memory, triggers the accelerator, and reads
   the int8 output back. Keep the signature identical.
2. Re-run `python -m fpga_tests.test_layer --layer conv_b --tol-lsb 0`.
3. The HW-vs-Sim comparison should remain at 0 LSB (bit-identical).
   If not, the saved `.npz` from a passing C-backend run is the replay
   vector you need to debug the FPGA's requantizer.

That is exactly the "freeze one layer at a time" discipline from §10.2 of
`fpga_flow.md`.

---

## 7. Reference numbers (seed 42, default configs)

Numbers you should see on the current build, for sanity:

| layer   | output shape            | Sim vs SW (int8 max) | HW vs Sim (int8 max) | SW vs float ref (rms) |
|---------|-------------------------|---------------------|-----------------------|-----------------------|
| conv_a  | (1, 54, 16, 56, 56)     | ≤ 2                 | 0                     | ~1e-2                 |
| conv_b  | (1, 54, 16, 28, 28)     | ≤ 1                 | 0                     | ~1e-2                 |
| conv_c  | (1, 24, 16, 28, 28)     | ≤ 1                 | 0                     | ~1e-2                 |
| conv_t  | (1, 24, 16, 112, 112)   | ≤ 1                 | 0                     | ~1e-2                 |
| conv_xy | (1, 24, 16, 112, 112)   | ≤ 1                 | 0                     | ~1e-2                 |

The "HW vs Sim = 0" column confirms that the C backend and the Python
simulator implement identical requantization logic.
