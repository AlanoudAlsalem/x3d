# FPGA offload bring-up: per-layer int8 conv tests

This folder is a scaffold for validating the int8 FPGA-offload path one
convolution at a time, with no real hardware connected yet. It implements
§7 ("tensor-by-tensor dataflow") and §10.4 ("dump int8 tensors at
boundaries") of `../fpga_flow.md`.

The first target is `conv_b` (the 3×3×3 depthwise inside every bottleneck),
but the harness is fully generic — `conv_a`, `conv_c`, `conv_t`, and
`conv_xy` are all supported out of the box.

---

## 1. What problem this solves

On the PolarFire SoC we want to move 3D convolutions off the RISC-V CPU and
onto the FPGA fabric. The FPGA will only ever see **int8** data. Before
wiring any real hardware, we need to answer two questions in software:

1. **Does our quantization math work?** Given a float32 input and a
   float32-trained conv, can we compute an int8 input, int8 weights, run an
   int8 conv, and get back a float32 output that is close to the original
   float32 result?
2. **Does the FPGA's fixed-point requantizer agree with the "obvious"
   float32 implementation?** The real hardware will use `(M0, n)` shift-and-
   multiply instead of float multiply. We need a harness that compares the
   two bit-for-bit per layer, so when the real FPGA lands we can tell
   whether a mismatch comes from bad hardware or from bad math.

This folder answers both. It runs the exact same conv layer four ways:

| path                      | where it runs | scale math   | requantize | purpose                        |
|---------------------------|--------------|--------------|------------|--------------------------------|
| float reference           | CPU          | none          | none       | accuracy ceiling              |
| software int8 (gold)      | CPU          | float32 `M`  | float      | "correct" int8 output          |
| FPGA simulator            | CPU          | int `M0,n`   | fixed-pt   | stand-in for the real FPGA     |
| *(later)* real FPGA       | FPGA fabric  | int `M0,n`   | fixed-pt   | must match the sim bit-for-bit |

---

## 2. How to run it

```bash
# defaults: --layer conv_b --seed 42 --out-dir fpga_tests/runs --tol-lsb 2
python -m fpga_tests.test_layer

# test a different conv type
python -m fpga_tests.test_layer --layer conv_a
python -m fpga_tests.test_layer --layer conv_t --seed 7
```

Each run prints a one-screen summary and writes two files into
`fpga_tests/runs/`:

* `<layer>_seed<N>.npz` — every tensor, scale, and multiplier from the run
  (`x_f32`, `W`, `s_in`, `s_w`, `s_out`, `M`, `M0`, `n`, `x_q`, `W_q`,
  `y_ref_f32`, `y_sw_q`, `y_hw_q`, `y_sw_f32`, `y_hw_f32`).
* `<layer>_seed<N>.json` — a human-readable summary with all the diff
  statistics and a pass/fail flag.

Re-running with the same seed produces bit-identical results.

### Pass criterion

The script flags a run as **PASS** if the software-int8 output and the
FPGA-sim output disagree by at most `--tol-lsb` int8 levels at every
element. The default tolerance is 2 LSB, which absorbs the small ulp-level
disagreements between float rounding and fixed-point round-half-away-from-
zero. Once the software reference is switched to fixed-point as well
(per §5.3 of `fpga_flow.md`), we will tighten this to exactly 0.

---

## 3. File layout

```
fpga_tests/
├── README.md            # this file
├── __init__.py
├── layer_configs.py     # one LayerConfig entry per conv type — the only
│                        # place you change to test a new layer
├── quant.py             # symmetric int8 quantize / dequantize / scales,
│                        # float-M and fixed-point (M0, n) requantization
├── kernels.py           # sw_int8_conv3d (ground truth) and
│                        # fpga_sim_int8_conv3d (FPGA stand-in)
├── test_layer.py        # CLI entry point. Builds layer, runs all four
│                        # paths, saves everything, prints diff stats
└── runs/                # output tensors + JSON summaries (gitignore candidate)
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
   * `s_out = max(|y_ref_f32|) / 127`        — per-tensor (bring-up shortcut;
     in production this comes from offline PTQ calibration, not from the
     reference output)
   * `M[c] = (s_in * s_w[c]) / s_out`        — float32 multiplier
   * `(M0[c], n[c])`                         — fixed-point form of `M[c]`
5. **Quantize** `x_f32 → x_q` (int8) and `W → W_q` (int8 per-channel).
6. **Run two int8 convs**:
   * *Software gold*: integer accumulate → `round(acc32 * M[c])` in float32
     → clip → int8. This is `sw_int8_conv3d`.
   * *FPGA sim*: integer accumulate → `(acc32 * M0[c]) >> n[c]` with
     rounding → clip → int8. This is `fpga_sim_int8_conv3d` and is the
     **one function that will get swapped out for a real DMA call** once
     the FPGA accelerator exists. Everything else stays identical.
7. **Dequantize** both int8 outputs by multiplying by `s_out`.
8. **Compare** three diffs and save all tensors to an `.npz`:
   * HW-sim vs SW (int8 space) — should be ≤ tolerance
   * SW int8 vs float reference — quantization-error budget
   * HW-sim int8 vs float reference — total error budget

### The integer-accumulate trick

The harness does not ship its own int8 conv kernel. Instead it reuses
`scratch.ops.conv3d.conv3d_forward`, passing int8 values as float32. The
float32 kernel happens to produce exact integer accumulators as long as the
intermediate sums fit in the 24-bit float mantissa — which they do for
every layer in X3D-M. We then `np.rint` back to int32. This keeps the
harness dependency-free and lets us focus on the quantization math rather
than on writing yet another conv kernel. When the real C/FPGA int8 kernel
lands, `_int_conv_accumulator` in `kernels.py` is the only thing that
changes.

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

When the FPGA int8 accelerator exists:

1. Implement a wrapper that takes `(x_q, W_q, M0, n, stride, padding,
   groups)`, DMAs into FPGA memory, triggers the accelerator, and reads
   the int8 output back.
2. Replace the body of `fpga_sim_int8_conv3d` in `kernels.py` with a call
   to that wrapper. Keep the signature identical.
3. Re-run `python -m fpga_tests.test_layer --layer conv_b --tol-lsb 0`.
4. If it doesn't pass at 0 LSB tolerance, the bug is in the FPGA
   requantizer's rounding mode or in an edge-case clip — the saved `.npz`
   from a passing software run is the replay vector you need to debug it.

That is exactly the "freeze one layer at a time" discipline from §10.2 of
`fpga_flow.md`.

---

## 7. Reference numbers (seed 42, default configs)

Numbers you should see on the current pure-CPU build, for sanity:

| layer   | output shape            | HW vs SW (int8 max) | SW vs float ref (rms) |
|---------|-------------------------|---------------------|-----------------------|
| conv_a  | (1, 54, 16, 56, 56)     | ≤ 2                 | ~1e-2                 |
| conv_b  | (1, 54, 16, 28, 28)     | ≤ 1                 | ~1e-2                 |
| conv_c  | (1, 24, 16, 28, 28)     | ≤ 1                 | ~1e-2                 |
| conv_t  | (1, 24, 16, 112, 112)   | ≤ 1                 | ~1e-2                 |
| conv_xy | (1, 24, 16, 112, 112)   | ≤ 1                 | ~1e-2                 |

These are not accuracy targets for the full model — they are sanity checks
that the per-layer quantization pipeline is wired correctly.
