"""
Three int8 conv "kernels" used by the bring-up harness:

  1. `sw_int8_conv3d`       — software reference. Runs in NumPy using the
                              existing float32 conv kernel from scratch.ops,
                              then requantizes with float32 M[c]. This is the
                              "gold-standard ground truth" we compare against.

  2. `fpga_sim_int8_conv3d` — Python stand-in for the FPGA. Same integer
                              accumulate, but requantization uses the
                              fixed-point (M0, n) form that the real FPGA
                              will use.

  3. `fpga_hw_int8_conv3d`  — C backend that mirrors the FPGA datapath:
                              native int8×int8 accumulation and (M0, n)
                              requantization in C with pthreads. This is the
                              function that gets swapped for a DMA call once
                              the real FPGA fabric is wired up.

All three share the same interface convention:
    (x_q, W_q, <requant params>, stride, padding, groups) -> int8 output

Integer accumulation trick (sw_int8 and fpga_sim paths)
-------------------------------------------------------
We pass int8 values to the existing float32 kernel as float32 and the output
is *exactly* the integer convolution result, as long as the accumulators fit
in the float32 mantissa (24 bits). For our layer shapes the worst case is
well under 2^24, so the conversion back to int32 is lossless. The C backend
(fpga_hw) does true int8×int8→int32 arithmetic natively.
"""

from __future__ import annotations
import numpy as np

from scratch.ops.conv3d import conv3d_forward
from scratch.ops.conv3d_fpga import fpga_hw_int8_conv3d, is_fpga_native_available
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
    """FPGA stand-in (Python): fixed-point (M0, n) requantize."""
    acc32 = _int_conv_accumulator(x_q, W_q, stride, padding, groups)
    return apply_requantize_fixed_point(acc32, M0, n)


# fpga_hw_int8_conv3d is imported from scratch.ops.conv3d_fpga above.
# It has the same signature as fpga_sim_int8_conv3d:
#   fpga_hw_int8_conv3d(x_q, W_q, M0, n, stride, padding, groups) -> int8
#
# Use is_fpga_native_available() to check if the C library is compiled.
