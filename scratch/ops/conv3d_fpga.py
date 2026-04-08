"""
Int8 3D convolution via the FPGA-offload C backend (libconv3d_fpga.so).

Wraps the C library that mirrors the FPGA accelerator's datapath:
  int8 input × int8 weights → int32 accumulate → (M0, n) requantize → int8 output

The C library is structurally identical to the real FPGA's compute path
(same accumulation, same fixed-point requantization), but runs on the CPU.
When the real FPGA fabric is wired up, the DMA layer replaces this call.

This module exposes a single function, `fpga_hw_int8_conv3d`, with the same
signature as `fpga_sim_int8_conv3d` in fpga_tests/kernels.py so they can be
compared directly in the test harness.
"""

from __future__ import annotations
import ctypes
import os
import numpy as np
from typing import Tuple, Union

_c_lib = None
_c_int8_p = ctypes.POINTER(ctypes.c_int8)
_c_int32_p = ctypes.POINTER(ctypes.c_int32)
_c_int64_p = ctypes.POINTER(ctypes.c_int64)


def _load_fpga_backend() -> None:
    global _c_lib
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv3d_fpga_c")

    for name in ("libconv3d_fpga.so", "libconv3d_fpga.dylib"):
        lib_path = os.path.join(lib_dir, name)
        if os.path.isfile(lib_path):
            break
    else:
        print("[conv3d_fpga] C backend not found — build with:  "
              "make -C scratch/ops/conv3d_fpga_c")
        return

    try:
        _c_lib = ctypes.CDLL(lib_path)
        _c_lib.conv3d_fpga_int8.restype = None
        _c_lib.conv3d_fpga_int8.argtypes = [
            _c_int8_p,                                      # input
            _c_int8_p,                                      # weight
            _c_int8_p,                                      # output
            _c_int64_p,                                     # M0
            _c_int32_p,                                     # n
            ctypes.c_int, ctypes.c_int,                     # B, C_in
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # T, H, W
            ctypes.c_int,                                   # C_out
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # kT, kH, kW
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # stride_t/h/w
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # pad_t/h/w
            ctypes.c_int,                                   # groups
        ]
        print(f"[conv3d_fpga] C backend loaded from {lib_path}")
    except OSError as e:
        print(f"[conv3d_fpga] C backend load failed: {e}")


_load_fpga_backend()


def is_fpga_native_available() -> bool:
    """Return True if the FPGA int8 C shared library is loaded."""
    return _c_lib is not None


def fpga_hw_int8_conv3d(
    x_q: np.ndarray,
    W_q: np.ndarray,
    M0: np.ndarray,
    n: np.ndarray,
    stride: Union[Tuple[int, int, int], list],
    padding: Union[Tuple[int, int, int], list],
    groups: int,
) -> np.ndarray:
    """
    Int8 conv3d via the FPGA-offload C backend.

    Signature matches fpga_sim_int8_conv3d in fpga_tests/kernels.py:
        (x_q, W_q, M0, n, stride, padding, groups) -> int8 output

    Parameters
    ----------
    x_q    : int8 input  (B, C_in, T, H, W)
    W_q    : int8 weight (C_out, C_in/groups, kT, kH, kW)
    M0     : int64 per-channel multiplier (C_out,)
    n      : int32 per-channel shift (C_out,)
    stride : (stride_t, stride_h, stride_w)
    padding: (pad_t, pad_h, pad_w)
    groups : 1=standard, C_in=depthwise
    """
    if _c_lib is None:
        raise RuntimeError(
            "FPGA int8 C backend not available. Build with:  "
            "make -C scratch/ops/conv3d_fpga_c"
        )

    x_q = np.ascontiguousarray(x_q, dtype=np.int8)
    W_q = np.ascontiguousarray(W_q, dtype=np.int8)
    M0 = np.ascontiguousarray(M0, dtype=np.int64)
    n = np.ascontiguousarray(n, dtype=np.int32)

    B, C_in, T, H, W = x_q.shape
    C_out, _, kT, kH, kW = W_q.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    T_out = (T + 2 * pt - kT) // st + 1
    H_out = (H + 2 * ph - kH) // sh + 1
    W_out = (W + 2 * pw - kW) // sw + 1

    out = np.empty((B, C_out, T_out, H_out, W_out), dtype=np.int8)

    _c_lib.conv3d_fpga_int8(
        x_q.ctypes.data_as(_c_int8_p),
        W_q.ctypes.data_as(_c_int8_p),
        out.ctypes.data_as(_c_int8_p),
        M0.ctypes.data_as(_c_int64_p),
        n.ctypes.data_as(_c_int32_p),
        B, C_in, T, H, W,
        C_out, kT, kH, kW,
        st, sh, sw,
        pt, ph, pw,
        groups,
    )

    return out
