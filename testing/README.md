# Minimal int8 3D Conv — Software vs FPGA Test

A tiny, self-contained C test harness for a 3D convolution with **int8**
inputs and weights and **int32** outputs. No quantization, no
requantization, no bias, no threading — just the raw convolution.

Two implementations share exactly the same API:

| function            | what it does                                           |
|---------------------|--------------------------------------------------------|
| `conv3d_int8_sw`    | Pure-C CPU reference. Nested loops. Ground truth.      |
| `conv3d_int8_fpga`  | FPGA-offload entry point. Today it is a stub with three clearly marked steps (DMA in → launch → DMA out); the real driver plugs in there. |

The `main.c` program generates a small deterministic int8 input and
weight tensor, runs both functions, and compares the outputs
element-by-element.

---

## Folder contents

```
testing/
├── README.md         # this file
├── conv3d_simple.h   # shared API declaration for both implementations
├── conv3d_sw.c       # software reference (nested loops)
├── conv3d_fpga.c     # FPGA entry point (stubbed DMA / launch / readback)
├── main.c            # test program: runs both, compares, prints result
└── Makefile          # gcc -O2 -Wall -Wextra -std=c11
```

No external dependencies. Only a C11 compiler (`gcc` or `clang`) and
`make`. No Python, no NumPy, no libraries.

---

## Build and run

```bash
cd testing
make           # compiles test_conv3d_simple
make run       # compiles and runs the test
make clean     # removes object files and the binary
```

Expected output on success:

```
input shape  : (1, 4, 4, 8, 8)
weight shape : (4, 1, 3, 3, 3)
output shape : (1, 4, 4, 8, 8)
groups       : 4

conv3d_int8_sw   returned 0
conv3d_int8_fpga returned 0

total output elements : 1024
mismatched elements   : 0
max abs diff          : 0

first 8 output values (sw vs fpga):
  [0]  sw=11640     fpga=11640     ok
  ...
PASS: outputs match bit-for-bit.
```

Exit code is `0` on PASS and `1` on FAIL so you can wire it into CI.

---

## The API

Both functions are declared in `conv3d_simple.h` with **identical**
signatures:

```c
int conv3d_int8_sw  (const int8_t *input,
                     const int8_t *weight,
                     int32_t      *output,
                     int B,  int C_in, int T, int H, int W,
                     int C_out, int kT, int kH, int kW,
                     int stride_t, int stride_h, int stride_w,
                     int pad_t,    int pad_h,    int pad_w,
                     int groups);

int conv3d_int8_fpga(/* …exactly the same parameters… */);
```

### Tensor layouts (all contiguous, NCTHW)

| pointer   | dtype  | shape                                             |
|-----------|--------|---------------------------------------------------|
| `input`   | int8   | `[B, C_in, T, H, W]`                              |
| `weight`  | int8   | `[C_out, C_in/groups, kT, kH, kW]`                |
| `output`  | int32  | `[B, C_out, T_out, H_out, W_out]` (caller-allocated) |

With:

```
T_out = (T + 2*pad_t - kT) / stride_t + 1
H_out = (H + 2*pad_h - kH) / stride_h + 1
W_out = (W + 2*pad_w - kW) / stride_w + 1
```

`groups = 1` is a standard convolution. `groups = C_in = C_out` is a
depthwise convolution (the default in the test).

### Return codes

| code | meaning                                                       |
|------|---------------------------------------------------------------|
|  `0` | success                                                       |
| `-1` | invalid argument (NULL pointer, bad `groups`, degenerate shape)|
| `-2` | FPGA offload failed (only returned by `conv3d_int8_fpga`)     |

---

## How the FPGA stub is structured

`conv3d_fpga.c` is organised as the three phases a real driver will
have, each in its own helper function so the replacement points are
obvious:

```c
/* STEP 1 — DMA input + weights to FPGA-visible memory */
fpga_dma_to_device(input,  in_bytes);
fpga_dma_to_device(weight, w_bytes);

/* STEP 2 — Kick off the FPGA accelerator, wait for done */
fpga_launch_and_wait(shape, stride, padding, groups);

/* STEP 3 — DMA the int32 output back to the host buffer */
fpga_dma_from_device(output, out_bytes);
```

Today those three helpers are no-op stubs with `TODO` comments, and the
actual output is filled in by calling `conv3d_int8_sw` during STEP 2.
This is deliberate: it guarantees the compare harness passes today
(`sw == fpga`) so the full plumbing is exercised end-to-end without any
hardware in the loop.

### When the real FPGA fabric is ready

1. Implement the three helpers in `conv3d_fpga.c`:
   - `fpga_dma_to_device`   — memcpy to the DMA-mapped buffer, flush cache.
   - `fpga_launch_and_wait` — write shape/stride control registers, set the
     `start` bit, poll `done`.
   - `fpga_dma_from_device` — invalidate cache, memcpy from the DMA buffer.
2. Delete the `conv3d_int8_sw(...)` call inside `conv3d_int8_fpga` that
   is currently standing in for the hardware.
3. Run `make run`. The software reference is still computed in parallel
   and compared against the hardware output; any mismatch surfaces
   immediately.

Nothing else — not the header, not `main.c`, not the Makefile — needs
to change.

---

## Changing the test shape

Open `main.c` and edit the constants at the top of `main()`:

```c
const int B      = 1;
const int C_in   = 4;
const int T      = 4;
const int H      = 8;
const int W      = 8;
const int C_out  = 4;
const int kT = 3, kH = 3, kW = 3;
const int stride_t = 1, stride_h = 1, stride_w = 1;
const int pad_t    = 1, pad_h    = 1, pad_w    = 1;
const int groups   = 4;   /* depthwise */
```

The test data is generated by a deterministic LCG PRNG (`rand_int8`) so
runs are reproducible across machines and compilers.

---

## Packaging for sharing

The folder is self-contained. To send it to someone:

```bash
tar czf testing.tar.gz testing/
# or
zip -r testing.zip testing/
```

They only need `gcc` (or `clang`) and `make` to build and run.
