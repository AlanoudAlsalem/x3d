# X3D-M on PolarFire SoC: Data Flow, Quantization, and FPGA Offload

This document explains, from first principles, how data moves between the RISC-V
CPU (the four U54 cores) and the FPGA fabric on the PolarFire SoC Icicle Kit
when running X3D-M inference. It covers two execution paths — the pure float32
path and the hybrid int8 path — with every formula written out explicitly.

The goal is that after reading this document you can point at any piece of data
in the pipeline and say exactly what numeric type it is, what scale factor it
carries, and which processor is touching it at that moment.

---

## Table of Contents

1. [Background: what lives where on the PolarFire SoC](#1-background)
2. [Path A: Pure float32 (baseline)](#2-path-a-pure-float32)
3. [Path B: Hybrid int8 with float32 glue](#3-path-b-hybrid-int8)
4. [The quantization formulas in full](#4-quantization-formulas)
5. [The requantization formula and why it is fixed-point on hardware](#5-requantization)
6. [BatchNorm folding and why it happens before anything else](#6-bn-folding)
7. [Tensor-by-tensor dataflow through a single conv layer](#7-tensor-by-tensor)
8. [End-to-end example: Stem conv through first ResBlock](#8-end-to-end)
9. [Error budget: where precision is lost](#9-error-budget)
10. [Practical checklist for bring-up and debugging](#10-checklist)

---

## 1. Background

### 1.1 The players

- **RISC-V CPU (4× U54 cores)**. This is where your existing `scratch` library
  runs. All tensors it touches are NumPy `float32` arrays laid out as
  `(B, C, T, H, W)`. It has 2 GB of LPDDR4 DRAM and a small L1/L2 cache.
- **FPGA fabric**. This is the programmable logic on the same chip. It will
  eventually contain a custom int8 3D-convolution accelerator. Until that is
  built, we *simulate* the FPGA in software using a bit-accurate reference
  kernel so the rest of the pipeline can be developed and tested.
- **Shared DRAM**. The CPU and the FPGA both access the same LPDDR4 memory.
  Moving data between them is not a PCIe-style copy; it is a DMA transfer
  inside the same physical memory, so the cost is dominated by memory
  bandwidth, not latency.

### 1.2 Why this split exists

For this phase of the project we only accelerate **3D convolutions** on the
FPGA. Every other operation — BatchNorm (already folded), ReLU, SiLU,
Squeeze-Excitation, skip-connection adds, average pooling, the final Linear
head — stays on the RISC-V CPU in float32. This choice is deliberate:

- Convolution is ~85% of the total compute in X3D-M. Accelerating just
  convolution captures almost all of the available speedup.
- Keeping non-conv ops on the CPU avoids having to quantize non-linearities
  like SiLU, which are awkward in integer hardware.
- It isolates the FPGA work: the FPGA has one interface to implement (int8
  conv) and nothing else. Every other op is untouched by the hardware
  project.

The tradeoff is that at every conv boundary we have to quantize (float32 → int8)
on the way in and dequantize (int8 → float32) on the way out. Those conversions
are cheap — one multiply per element — and run on the CPU in parallel with the
FPGA's work.

---

## 2. Path A: Pure float32

This is the baseline. Everything is float32, nothing is quantized, nothing goes
near the FPGA. It exists for three reasons: correctness reference, accuracy
ceiling, and CPU-only performance benchmark.

```
  [video frames] --(preprocess)--> float32 tensor (1, 3, 16, 224, 224)
         │
         ▼
   ┌────────────┐
   │    Stem    │   float32 in  → float32 out   (all on CPU)
   └────┬───────┘
        ▼
   ┌────────────┐
   │  Stage 2   │   every Conv3d calls scratch.ops.conv3d.conv3d_forward
   └────┬───────┘   which dispatches to {slow, fast, native} float32 kernels
        ▼
   ┌────────────┐
   │  Stage 3   │   float32 throughout
   └────┬───────┘
        ▼
   ...
        ▼
   ┌────────────┐
   │    Head    │   float32 Linear → logits (1, 400)
   └────────────┘
```

In this path, the FPGA is completely idle. There is no quantization, no int8,
no scale factors. This is the path you run today when you invoke `main.py`
without any quantization flags.

The model lives in `scratch/models/x3d_m.py` and is loaded with float32
weights from `weights/x3d_m_kinetics400.npz` via
`scratch.load_weights.load_pretrained_numpy`.

**Module pairing**: `main.py` + `scratch/*` + float32 `.npz`.

---

## 3. Path B: Hybrid int8

This is the path you will run on the FPGA. The model is structurally identical
to Path A, but every Conv3d is replaced with a **quantized Conv3d** that knows
how to quantize its input, dispatch to the FPGA (or the software reference),
and dequantize its output. Every non-conv op remains float32.

```
  [video frames] --(preprocess)--> float32 tensor (1, 3, 16, 224, 224)
         │
         ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ QuantizedConv3d (Stem conv_t)                                 │
   │                                                                │
   │   float32 in                                                   │
   │      │                                                         │
   │      ▼   (1) QUANTIZE ON CPU                                   │
   │   int8 activation + input_scale (scalar)                       │
   │      │                                                         │
   │      ▼   (2) DMA TO FPGA (or call software int8 kernel)        │
   │   ┌──────────────────────────────────────────────────┐         │
   │   │  FPGA accelerator                                │         │
   │   │    int8 × int8 → int32 accumulate                │         │
   │   │    + int32 bias                                   │         │
   │   │    REQUANTIZE using M[c]                          │         │
   │   │  int8 output                                      │         │
   │   └──────────────────────────────────────────────────┘         │
   │      │                                                         │
   │      ▼   (3) DMA BACK + DEQUANTIZE ON CPU                      │
   │   float32 output                                               │
   └────┬───────────────────────────────────────────────────────────┘
        ▼
   BatchNorm3d (already folded away → nn.Identity)
        ▼
   ReLU       ← float32 CPU op from scratch/ops/activations.py
        ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ QuantizedConv3d (next conv) — same pattern                     │
   └────┬───────────────────────────────────────────────────────────┘
        ▼
   ...
```

Key points about this path:

- The float32 tensor between two conv layers **does pass through the CPU** in
  float32 form. It is not kept as int8 across layer boundaries. This is a
  deliberate simplification: it makes the hybrid model behave, from the outside,
  exactly like the float32 model, just with slightly different numbers. All the
  non-conv ops can stay unchanged.
- Every QuantizedConv3d carries its own `input_scale` and `output_scale` as
  precomputed constants from the PTQ calibration. These never change at
  runtime.
- The FPGA never touches float32 data. It sees only int8 inputs, int8 weights,
  int32 biases, and an int16 or int32 requantization multiplier (see §5).
  Everything crossing the CPU↔FPGA boundary is already integer.

**Module pairing**: `main_int8.py` + `scratch.quantized.*` + int8 `.npz`.

---

## 4. Quantization formulas

Every formula below assumes **symmetric quantization**, meaning the integer 0
always represents the real value 0.0 and the range is balanced around zero.
This is the scheme we chose because it eliminates zero-point subtractions from
the FPGA hot path.

### 4.1 Symmetric int8 range

Int8 is signed 8-bit, so the raw range is `[-128, +127]`. We deliberately
reserve `-128` and use only `[-127, +127]`, a symmetric 255-value range. This
avoids an asymmetry that complicates fixed-point multiplication and costs
essentially nothing in accuracy.

### 4.2 Quantizing a tensor (float32 → int8)

Given a float32 tensor `x` and a precomputed scale `s > 0`:

```
q = clip( round( x / s ),  -127,  +127 ).astype(int8)
```

To recover an approximation of `x`:

```
x' = q.astype(float32) * s        (≈ x, with quantization error)
```

The scale `s` has type float32 on the CPU side and is baked into fixed-point
constants on the FPGA side (see §5). For activations, `s` is a single scalar
per tensor. For weights, `s` is a vector with one entry per output channel —
this is called **per-channel** quantization.

### 4.3 Choosing the scale

For **weights**, `s` is determined once during PTQ by looking at the absolute
maximum of each output channel:

```
for each output channel c:
    s_w[c] = max( |W[c, :, :, :, :]| ) / 127
```

For **activations**, `s` is determined by running the float32 model on a small
calibration set of representative inputs and recording the absolute maximum
seen at each tensor location:

```
s_a = max_over_calibration_batches( |a| ) / 127
```

### 4.4 The weight scale is per-channel, the activation scale is per-tensor

This is the single most important asymmetry to remember:

- **Weights**: one scale per output channel. This is because weights are fixed
  constants — the FPGA can store a small table of scales and apply the right
  one per channel at no runtime cost. Per-channel is much more accurate than
  per-tensor for convolution weights, which often have very different
  magnitudes across channels.
- **Activations**: one scale for the entire tensor. This keeps the data stream
  uniform — every element of an activation tensor uses the same scale, so the
  FPGA doesn't need a dynamic scale lookup during the quantize step.

### 4.5 Bias quantization

Biases are added directly into the int32 accumulator during convolution, so
they need to live in "int32 accumulator space". The correct scale is:

```
s_b[c] = s_in * s_w[c]
```

where `s_in` is the input activation scale and `s_w[c]` is the per-channel
weight scale. The quantized bias is then:

```
b_q[c] = round( b[c] / s_b[c] ).astype(int32)
```

No clipping is needed in practice because int32 has more than enough range to
hold a scaled bias.

---

## 5. Requantization

After a convolution, we have an int32 accumulator that represents a value in
"accumulator units" — its scale is `s_in * s_w[c]`. The next layer expects
int8 with scale `s_out`. We need to convert:

```
            acc32[c] * (s_in * s_w[c])
out_int8 = ─────────────────────────────   clipped to [-127, +127]
                    s_out
```

Rewriting with `M[c] = (s_in * s_w[c]) / s_out`:

```
out_int8 = clip( round( acc32[c] * M[c] ), -127, +127 ).astype(int8)
```

### 5.1 Why `M[c]` is always in `(0, 1)` for well-calibrated models

If activations don't explode between layers — which is true for any reasonable
pretrained model — `s_out` is roughly the same order of magnitude as `s_in`,
and `s_w[c]` is small (on the order of 0.01). So `M[c]` is typically in the
range `[1e-4, 1e-1]`. This matters because a multiplier less than 1 can be
implemented very cheaply in fixed-point hardware.

### 5.2 Fixed-point implementation on the FPGA

The FPGA does not do floating-point multiplication. Instead, for each channel
we represent `M[c]` as:

```
M[c] ≈ M0[c] * 2^(-n[c])
```

where `M0[c]` is an int32 (or int16) in the range `[2^30, 2^31 - 1]` and
`n[c]` is a small positive integer shift. The requantization then becomes:

```
out_int8 = clip(
    round_to_nearest_even(
        (acc32[c] * M0[c]) >> n[c]
    ),
    -127, +127
).astype(int8)
```

Both operations — the int32×int32 multiply producing int64, and the right shift
with rounding — are cheap in FPGA logic. The per-channel table `(M0[c], n[c])`
is precomputed once on the CPU at model load time and written into the FPGA's
constant memory, next to the int8 weights.

This is the same fixed-point requantization used by TFLite, QNNPACK, and
CMSIS-NN. It is the universal recipe.

### 5.3 Where the software reference kernel does it

In the software reference path (the NumPy kernel in
`scratch/quantized/conv3d_int8.py`), we cheat slightly and use float32 for
`M[c]` because NumPy's float math is fast on the development machine and we
don't care about matching the FPGA bit-for-bit yet. Once the FPGA exists, the
software reference will be switched to the fixed-point form so its outputs are
**bit-identical** to the FPGA's, which is what you want for differential
debugging.

---

## 6. BatchNorm folding

Every BatchNorm3d in the model is folded into the preceding Conv3d **before**
quantization. After folding, each BN layer becomes the identity and has no
runtime cost.

### 6.1 The math

BatchNorm at inference time, per channel:

```
y[c] = gamma[c] * (x[c] - mean[c]) / sqrt(var[c] + eps) + beta[c]
```

Define `k[c] = gamma[c] / sqrt(var[c] + eps)`. Then BN is equivalent to:

```
y[c] = k[c] * x[c] + (beta[c] - k[c] * mean[c])
```

A preceding Conv3d computes:

```
x[c] = sum_{c_in, t, h, w} W[c, c_in, t, h, w] * input + b[c]
```

Because BN is a per-channel affine transform and convolution is linear, we can
absorb BN into the conv weights and bias:

```
W'[c, c_in, t, h, w] = k[c] * W[c, c_in, t, h, w]
b'[c]                = k[c] * b[c] + (beta[c] - k[c] * mean[c])
                     = k[c] * (b[c] - mean[c]) + beta[c]
```

After folding, applying Conv3d with `(W', b')` gives exactly the same result
as the original `Conv3d → BN` sequence, up to floating-point rounding.

### 6.2 Why folding must happen before quantization

Two reasons:

1. **Hardware simplicity**: after folding, the FPGA only needs to implement
   int8 convolution. It does not need a separate BN stage or any per-channel
   affine after the conv. The requantization step at the end of the conv
   already handles the per-channel scaling that BN used to do.
2. **Quantization accuracy**: the folded weights have a smoother per-channel
   distribution than the unfolded weights, because the `gamma / sqrt(var+eps)`
   factor redistributes magnitude across channels. This makes per-channel
   weight quantization tighter and reduces quantization error.

The folding is performed once, in the PTQ script, and the exported `.npz`
contains folded weights. The runtime never sees unfolded BN.

---

## 7. Tensor-by-tensor dataflow through a single conv layer

This is the most concrete description. Walk through it once and everything
else falls into place.

### 7.1 Setup

Assume we are executing some QuantizedConv3d layer called `L` with these
precomputed constants, loaded from the int8 `.npz`:

| Name           | Dtype       | Shape                     | Meaning                          |
|----------------|-------------|---------------------------|----------------------------------|
| `W_q`          | `int8`      | `(O, I/g, kT, kH, kW)`    | Quantized weights                |
| `s_w`          | `float32`   | `(O,)`                    | Per-channel weight scales        |
| `b_q`          | `int32`     | `(O,)` or `None`          | Quantized bias                   |
| `s_in`         | `float32`   | `()` (scalar)             | Input activation scale           |
| `s_out`        | `float32`   | `()` (scalar)             | Output activation scale          |

Where `O = out_channels`, `I = in_channels`, `g = groups`. The per-channel
requantization multiplier is:

```
M[c] = (s_in * s_w[c]) / s_out            for c in [0, O)
```

`M` is a float32 vector of length `O`, precomputed once when the layer is
built.

### 7.2 Step-by-step

Let `x_f32` be the float32 tensor arriving at layer `L` from the previous op.

**Step 1: CPU quantizes the input.**

```
x_q = clip( round( x_f32 / s_in ), -127, +127 ).astype(int8)
```

This runs on the RISC-V CPU in NumPy today, or in a small C kernel on the SoC
tomorrow. It touches every element of `x_f32` once and is memory-bound. Cost
is negligible compared to the conv.

**Step 2: DMA to FPGA (or call software reference).**

The CPU hands the following pointers to the FPGA:

- `x_q` (int8 input activation tensor)
- `W_q` (int8 weights) — this is usually already resident in FPGA memory
  because weights don't change
- `b_q` (int32 bias) — ditto
- `M[c]` table — ditto

The DMA transfer on the Icicle Kit is a memory-bandwidth-limited operation;
the FPGA doesn't actually copy anything if it shares the DRAM with the CPU.

**Step 3: FPGA computes the accumulator.**

For each output position `(b, c, t, h, w)`:

```
acc32[b, c, t, h, w] =
    sum over (c_in, kt, kh, kw) of
        W_q[c, c_in, kt, kh, kw] * x_q[b, c_in*g, t_in, h_in, w_in]
```

(with appropriate stride, padding, and grouping; identical to the float32
conv, just with int8 inputs and an int32 output).

**Step 4: FPGA adds the bias.**

```
acc32[b, c, t, h, w] += b_q[c]           (if bias is present)
```

**Step 5: FPGA requantizes back to int8.**

```
y_q[b, c, t, h, w] = clip(
    round( acc32[b, c, t, h, w] * M[c] ),
    -127, +127
).astype(int8)
```

In the real FPGA this is the fixed-point `(M0, n)` multiply-shift from §5.2.
In the software reference it's a float32 multiply.

**Step 6: FPGA hands `y_q` back to the CPU.**

Again, no real copy — just a handoff of a memory region.

**Step 7: CPU dequantizes the output.**

```
y_f32 = y_q.astype(float32) * s_out
```

`y_f32` is the float32 tensor handed to the next op (BN-folded-away, then ReLU,
then skip-add, then the next conv). From the perspective of everything outside
the conv layer, it is as if a slightly-lossy float32 convolution just ran.

### 7.3 Dtype summary at each point

```
  CPU:      x_f32     float32       (B,  I, T, H, W)
  CPU:      x_q       int8          (B,  I, T, H, W)    ── quantize step
  → FPGA
  FPGA:     x_q       int8          (B,  I, T, H, W)
  FPGA:     W_q       int8          (O, I/g, kT, kH, kW)
  FPGA:     acc32     int32         (B, O, T', H', W')
  FPGA:     acc32+b   int32         (B, O, T', H', W')
  FPGA:     y_q       int8          (B, O, T', H', W')  ── requantize step
  ← CPU
  CPU:      y_q       int8          (B, O, T', H', W')
  CPU:      y_f32     float32       (B, O, T', H', W')  ── dequantize step
```

Note that the only float32 tensors in this path live at the two CPU boundaries.
Everything inside the conv is integer.

---

## 8. End-to-end example: Stem conv through first ResBlock

To make §7 concrete, here is what happens for a real early chunk of X3D-M.

```
Input video:   float32 (1, 3, 16, 224, 224)     [on CPU]

QuantizedConv3d  blocks.0.conv.conv_t   (1×3×3, 3→24)
    CPU:  quantize  float32 → int8 with input_scale_0
    FPGA: int8 conv → int32 → +bias → requantize → int8
    CPU:  dequantize int8 → float32 with output_scale_0

BatchNorm3d  blocks.0.norm        [Identity — folded into conv_t]

ReLU                                              [CPU, float32]

QuantizedConv3d  blocks.0.conv.conv_xy  (5×1×1 depthwise, 24→24)
    CPU:  quantize  float32 → int8 with input_scale_1
    FPGA: int8 depthwise conv → int32 → +bias → requantize → int8
    CPU:  dequantize int8 → float32 with output_scale_1

BatchNorm3d  blocks.0.norm        [Identity — folded]

ReLU                                              [CPU, float32]

---- entering blocks.1 (Stage 2 first ResBlock) ----

Skip branch:
    branch1 = QuantizedConv3d  blocks.1.res_blocks.0.branch1.conv  (1×1×1, 24→24, stride 2)
    Same quantize / FPGA / dequantize pattern as above.

Main branch:
    branch2.conv_a  QuantizedConv3d (1×1×1, 24→54)
    branch2.norm_a  Identity (folded BN)
    act_a           ReLU                       [CPU, float32]

    branch2.conv_b  QuantizedConv3d (3×3×3 depthwise, 54→54, stride 2)
    branch2.norm_b  Identity (folded BN)
    SE block                                    [CPU, float32]
    act_b           SiLU                        [CPU, float32]

    branch2.conv_c  QuantizedConv3d (1×1×1, 54→24)
    branch2.norm_c  Identity (folded BN)

Add:  main + skip                               [CPU, float32]
ReLU                                            [CPU, float32]
```

Things to notice:

- Every Conv3d becomes a quantize→FPGA→dequantize triple.
- Everything else stays float32 on the CPU.
- The SE block is entirely float32 — it contains a global pool, two 1×1×1
  convolutions, a sigmoid, and a multiplication. In Phase 1 the SE's internal
  convs are *also* left as float32, because they are tiny (less than 0.1% of
  total FLOPs) and not worth the engineering effort of offloading. This is
  configurable in `scratch/quantized/model.py`.
- Skip-connection addition happens in float32 on the CPU. After both branches
  have been dequantized back to float32, it is a simple NumPy add.

---

## 9. Error budget

Where does accuracy loss come from?

1. **Weight quantization** (small): per-channel symmetric int8 typically loses
   less than 0.5% top-1 accuracy on X3D-M after BN folding.
2. **Activation quantization** (moderate): per-tensor symmetric int8 is the
   biggest source of loss, around 1–1.5% top-1. This is why activation
   calibration matters: bad scales compound this.
3. **Requantization rounding** (negligible): round-to-nearest-even in
   fixed-point typically contributes less than 0.1%.
4. **Float→int8 boundary crossings at every conv** (moderate): because we
   dequantize between layers in this hybrid design, each conv's output scale
   is rounded twice — once on the way out of that conv, and once on the way
   into the next conv. Keeping the data int8 across the boundary would avoid
   this, and is a future optimization.

A reasonable target for this design is **float32 minus 2% top-1 on
Kinetics-400**. If PTQ drops more than that, the next step is quantization-aware
training, not further PTQ tweaking.

---

## 10. Checklist for bring-up and debugging

When you first wire the int8 path to a real FPGA, the following discipline
will save days of debugging.

1. **Validate the software reference first**. Run the int8 path with the
   software kernel (`scratch/quantized/conv3d_int8.py`) and compare its
   per-layer outputs against the float32 path. Expect small differences, not
   identical numbers. If they are wildly different, the bug is in
   quantization, not in the hardware.
2. **Freeze one layer at a time**. When swapping in the FPGA, start by
   routing a *single* conv layer through the FPGA and leave every other conv
   on the software reference. Verify that layer's outputs match bit-for-bit.
   Then add a second layer. Incremental enablement is the only sane way to
   debug a hardware accelerator.
3. **Bit-identical software and hardware**. Once the fixed-point
   requantization is implemented in both, the software reference and the
   FPGA should produce **identical** int8 outputs for the same int8 inputs.
   If they don't, it is almost always a rounding-mode mismatch or a missing
   clip in one of the two.
4. **Dump int8 tensors at boundaries**. Log `x_q`, `W_q`, `b_q`, `M[c]`, and
   `y_q` for each layer to disk (NumPy save). When something goes wrong, you
   can load these into the software reference and replay the FPGA's
   computation exactly.
5. **Check scale loading**. A surprisingly common bug is loading the wrong
   `input_scale` or `output_scale` because of a module-path typo. Verify that
   every QuantizedConv3d has non-default scales after model build.
6. **Never mix paths**. Do not try to run Path A and Path B in the same
   process on the same model object. Build two separate model instances —
   `X3D_M` and `QuantizedX3D_M` — and run them on separate inputs with
   separate entry points (`main.py` and `main_int8.py`). This is why the
   int8 integration is kept in a separate `scratch.quantized` subpackage.

---

## Appendix A: Glossary of symbols

| Symbol      | Type      | Meaning                                                  |
|-------------|-----------|----------------------------------------------------------|
| `x_f32`     | float32   | A float32 activation tensor on the CPU                   |
| `x_q`       | int8      | The same tensor after quantization                       |
| `W_q`       | int8      | Quantized conv weights                                   |
| `b_q`       | int32     | Quantized conv bias                                      |
| `s_in`      | float32   | Per-tensor input activation scale                        |
| `s_out`     | float32   | Per-tensor output activation scale                       |
| `s_w[c]`    | float32   | Per-channel weight scale                                 |
| `s_b[c]`    | float32   | Per-channel bias scale, equals `s_in * s_w[c]`           |
| `M[c]`      | float32   | Requantization multiplier `(s_in * s_w[c]) / s_out`      |
| `M0[c]`     | int32     | Fixed-point mantissa for `M[c]` on hardware              |
| `n[c]`      | int       | Fixed-point right-shift for `M[c]` on hardware           |
| `acc32`     | int32     | Convolution accumulator before requantization            |

## Appendix B: Which file does what

| File                                              | Path       | Purpose                                           |
|---------------------------------------------------|------------|---------------------------------------------------|
| `scratch/models/x3d_m.py`                         | Path A     | Float32 X3D-M model                               |
| `scratch/load_weights.py`                         | Path A     | Float32 weight loader                             |
| `scratch/ops/conv3d.py`                           | Path A     | Float32 conv kernels (slow/fast/native)           |
| `main.py`                                         | Path A     | Float32 inference entry point                     |
| `scripts/quantize_x3d_ptq.py`                     | Both       | Offline PTQ, produces int8 `.npz`                 |
| `scratch/quantized/conv3d_int8.py`                | Path B     | Software reference int8 conv kernel (FPGA proxy)  |
| `scratch/quantized/layers.py`                     | Path B     | QuantizedConv3d module                            |
| `scratch/quantized/load_int8_weights.py`          | Path B     | Int8 weight + scale loader                        |
| `scratch/quantized/model.py`                      | Path B     | QuantizedX3D_M model builder                      |
| `main_int8.py`                                    | Path B     | Int8 inference entry point                        |

The two paths share nothing at runtime. You can delete all of Path B and
Path A still works, and vice versa.
