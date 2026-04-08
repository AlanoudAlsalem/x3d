"""
Two int8 conv "kernels" used by the bring-up harness:

  1. `sw_int8_conv3d`      — software reference. Runs in NumPy using the
                             existing float32 conv kernel from scratch.ops,
                             then requantizes with float32 M[c]. This is the
                             "gold-standard ground truth" we compare against.

  2. `fpga_sim_int8_conv3d` — stand-in for the FPGA. Same integer accumulate,
                             but requantization uses the fixed-point (M0, n)
                             form that the real FPGA will use. Once the real
                             accelerator exists, this function is the ONE
                             place that gets swapped for a DMA call.

Integer accumulation trick
--------------------------
We do not (yet) have an int8 conv kernel in `scratch.ops`. We don't need one
for bring-up: if we pass int8 values to the existing float32 kernel as
float32, the output is *exactly* the integer convolution result, as long as
the accumulators fit in the float32 mantissa (24 bits). For our layer shapes
the worst case is well under 2^24, so the conversion back to int32 is lossless.
When the C / FPGA int8 path lands, this function gets replaced; the harness
above stays identical.
"""

from __future__ import annotations
import numpy as np

from scratch.ops.conv3d import conv3d_forward
from fpga_tests.quant import (
    apply_requantize_float,
    apply_requantize_fixed_point,
)


def _int_conv_accumulator(
    x_q: np.ndarray,
    W_q: np.ndarray,
    stride,
    padding,
    groups: int,
) -> np.ndarray:
    """
    Compute the int32 conv accumulator by running the existing float32 kernel
    on int8 values cast to float32. No bias. Result is exact.
    """
    x_f = x_q.astype(np.float32)
    W_f = W_q.astype(np.float32)
    acc_f = conv3d_forward(x_f, W_f, None, stride, padding, groups)
    # Accumulator is integer-valued; cast back.
    acc32 = np.rint(acc_f).astype(np.int32)
    return acc32


def sw_int8_conv3d(
    x_q: np.ndarray,
    W_q: np.ndarray,
    M: np.ndarray,
    stride,
    padding,
    groups: int,
) -> np.ndarray:
    """Software reference: float32 requantize."""
    acc32 = _int_conv_accumulator(x_q, W_q, stride, padding, groups)
    return apply_requantize_float(acc32, M)


def fpga_sim_int8_conv3d(
    x_q: np.ndarray,
    W_q: np.ndarray,
    M0: np.ndarray,
    n: np.ndarray,
    stride,
    padding,
    groups: int,
) -> np.ndarray:
    """FPGA stand-in: fixed-point (M0, n) requantize. Swap for a DMA call later."""
    acc32 = _int_conv_accumulator(x_q, W_q, stride, padding, groups)
    return apply_requantize_fixed_point(acc32, M0, n)
