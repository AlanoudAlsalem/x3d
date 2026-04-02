"""
3D convolution (no PyTorch). Supports standard and depthwise (groups=in_channels).

Four implementation strategies selectable via set_conv3d_method() or per-call:

  "slow"     — Pure NumPy fallback for platforms without OpenCV (e.g. some
               RISC-V embedded systems). No external dependencies beyond NumPy.

  "fast"     — Single-threaded OpenCV path. Uses cv2.filter2D for accelerated
               2-D convolution, looping over temporal/channel dimensions.

  "threaded" — Multi-threaded OpenCV path targeting the PolarFire SoC's 4 U54
               RISC-V application cores via adaptive hybrid parallelism
               (see DOCUMENTATION.md Section 14):
                 - Pointwise / standard convolutions: output-channel parallelism
                 - Depthwise convolutions: temporal parallelism within conv3d_core
               NumPy and OpenCV release the GIL during C-level computation, so
               Python threads achieve genuine parallelism for these workloads.

  "native"   — C shared-library backend (libconv3d.so) loaded via ctypes.
               Pthreads parallelism (4 threads), cache-friendly spatial tiling,
               separate fast paths for pointwise / depthwise / general conv.
               Build with: make -C scratch/ops/conv3d_c
               Falls back to RuntimeError if the library has not been compiled.

Default method: "fast"
"""

from __future__ import annotations
import ctypes
import os
import numpy as np
from typing import Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import cv2

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

VALID_METHODS = ("slow", "fast", "threaded", "native")
_default_method: str = "fast"

NUM_THREADS = 4  # PolarFire SoC: 4 U54 application cores
_thread_pool: Optional[ThreadPoolExecutor] = None

# ---------------------------------------------------------------------------
# C backend (native) — loaded once at import time
# ---------------------------------------------------------------------------

_c_lib = None
_c_float_p = ctypes.POINTER(ctypes.c_float)


def _load_c_backend() -> None:
    global _c_lib
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv3d_c")

    for name in ("libconv3d.so", "libconv3d.dylib"):
        lib_path = os.path.join(lib_dir, name)
        if os.path.isfile(lib_path):
            break
    else:
        print("[conv3d] C backend not found — build with:  make -C scratch/ops/conv3d_c")
        return

    try:
        _c_lib = ctypes.CDLL(lib_path)
        _c_lib.conv3d_forward_c.restype = None
        _c_lib.conv3d_forward_c.argtypes = [
            _c_float_p,                                     # input
            _c_float_p,                                     # weight
            _c_float_p,                                     # bias  (NULL ok)
            _c_float_p,                                     # output
            ctypes.c_int, ctypes.c_int,                     # B, C_in
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # T, H, W
            ctypes.c_int,                                   # C_out
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # kT, kH, kW
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # stride_t/h/w
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # pad_t/h/w
            ctypes.c_int,                                   # groups
        ]
        print(f"[conv3d] C backend loaded successfully from {lib_path}")
    except OSError as e:
        print(f"[conv3d] C backend load failed: {e}")


_load_c_backend()


def is_native_available() -> bool:
    """Return True if the C shared-library backend is loaded."""
    return _c_lib is not None


def _get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)
    return _thread_pool


def set_conv3d_method(method: str) -> None:
    """Set the default convolution implementation used by conv3d_forward.

    Args:
        method: One of "slow", "fast", "threaded", or "native".
    """
    global _default_method
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}, got {method!r}"
        )
    _default_method = method


def get_conv3d_method() -> str:
    """Return the current default convolution method."""
    return _default_method


def set_num_threads(n: int) -> None:
    """Change the number of worker threads (recreates the pool)."""
    global NUM_THREADS, _thread_pool
    if n < 1:
        raise ValueError("n must be >= 1")
    NUM_THREADS = n
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=False)
    _thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pad_3d(
    x: np.ndarray,
    pad_t: int, pad_h: int, pad_w: int,
) -> np.ndarray:
    """Zero-pad input on (T, H, W). x shape: (B, C, T, H, W)."""
    B, C, T, H, W = x.shape
    out = np.zeros((B, C, T + 2 * pad_t, H + 2 * pad_h, W + 2 * pad_w), dtype=x.dtype)
    out[:, :, pad_t : pad_t + T, pad_h : pad_h + H, pad_w : pad_w + W] = x
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def conv3d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
    method: Optional[str] = None,
) -> np.ndarray:
    """Dispatch to the selected 3D convolution implementation.

    Args:
        x:       Input  (B, C_in, T, H, W).
        weight:  Kernel (C_out, C_in//groups, kT, kH, kW).
        bias:    Optional bias (C_out,) or None.
        stride:  (stride_t, stride_h, stride_w).
        padding: (pad_t, pad_h, pad_w).
        groups:  1 = standard; C_in = depthwise.
        method:  Override the global default ("slow", "fast", "threaded", "native").
                 If None, uses the value set by set_conv3d_method().
    """
    if method is None:
        method = _default_method
    if method == "slow":
        return conv3d_forward_slow(x, weight, bias, stride, padding, groups)
    if method == "fast":
        return conv3d_forward_fast(x, weight, bias, stride, padding, groups)
    if method == "threaded":
        return conv3d_forward_threaded(x, weight, bias, stride, padding, groups)
    if method == "native":
        return conv3d_forward_native(x, weight, bias, stride, padding, groups)
    raise ValueError(
        f"Unknown conv3d method {method!r}; choose from {VALID_METHODS}"
    )


# ===================================================================
# METHOD 1: "slow" — pure NumPy (no OpenCV)
# ===================================================================

def conv3d_forward_slow(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """
    Compute 3D convolution. Data layout: (B, C, T, H, W).

    Args:
        x: Input tensor shape (B, in_channels, T, H, W).
        weight: Kernel shape (out_channels, in_channels//groups, kT, kH, kW).
        bias: Optional bias shape (out_channels,) or None.
        stride: (stride_t, stride_h, stride_w).
        padding: (pad_t, pad_h, pad_w).
        groups: Number of groups (1 = standard conv; in_channels = depthwise).

    Returns:
        Output tensor shape (B, out_channels, T', H', W').
    """
    B, in_c, T, H, W = x.shape
    out_c, c_per_group, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    if in_c % groups != 0 or out_c % groups != 0:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if in_c // groups != c_per_group:
        raise ValueError("weight in_channels per group mismatch")

    x_pad = _pad_3d(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1

    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=x.dtype)

    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group
            c_end = c_start + c_per_group
            acc = np.zeros((T_out, H_out, W_out), dtype=x.dtype)
            for c in range(c_per_group):
                w = weight[oc, c, :, :, :]  # (kT, kH, kW)
                inp = x_pad[b, c_start + c, :, :, :]  # (Tp, Hp, Wp)
                for tt in range(T_out):
                    for hh in range(H_out):
                        for ww in range(W_out):
                            t0, h0, w0 = tt * st, hh * sh, ww * sw
                            acc[tt, hh, ww] += np.sum(
                                inp[t0 : t0 + kT, h0 : h0 + kH, w0 : w0 + kW] * w
                            )
            out[b, oc, :, :, :] = acc
            if bias is not None:
                out[b, oc, :, :, :] += bias[oc]

    return out


# ===================================================================
# METHOD 2: "fast" — single-threaded OpenCV (cv2.filter2D)
# ===================================================================

def conv3d_forward_fast(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    B, in_c, T, H, W = x.shape
    out_c, c_per_group, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    x_pad = _pad_3d(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1
    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=x.dtype)

    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group
            c_end = c_start + c_per_group

            inp_volume = x_pad[b, c_start:c_end]
            kernel_volume = weight[oc]

            dense_out = conv3d_core(inp_volume, kernel_volume)

            strided_out = dense_out[::st, ::sh, ::sw]

            out[b, oc] = strided_out.astype(x.dtype)
            if bias is not None:
                out[b, oc] += bias[oc]

    return out


# ===================================================================
# METHOD 3: "threaded" — multi-threaded OpenCV with adaptive strategy
# ===================================================================

def conv3d_forward_threaded(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """Multi-threaded conv3d with adaptive hybrid parallelism.

    Pointwise / standard convolutions use output-channel parallelism
    (Strategy 1).  Depthwise convolutions use temporal parallelism
    within conv3d_core (Strategy 2).
    """
    B, in_c, T, H, W = x.shape
    out_c, c_per_group, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    x_pad = _pad_3d(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1
    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=x.dtype)

    is_pointwise = kT * kH * kW == 1
    is_depthwise = groups == out_c and groups > 1

    if is_pointwise or not is_depthwise:
        _conv3d_oc_parallel(x_pad, weight, bias, out,
                            B, out_c, groups, c_per_group, st, sh, sw)
    else:
        _conv3d_temporal_parallel(x_pad, weight, bias, out,
                                  B, out_c, groups, c_per_group, st, sh, sw)

    return out


def _conv3d_oc_parallel(
    x_pad: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    out: np.ndarray,
    B: int, out_c: int, groups: int, c_per_group: int,
    st: int, sh: int, sw: int,
) -> None:
    """Strategy 1: distribute (batch, output-channel) pairs across threads.

    Each thread processes a contiguous chunk of (b, oc) pairs.  Different
    chunks write to non-overlapping slices of *out*, so no synchronisation
    is needed.  conv3d_core spends nearly all time inside cv2.filter2D /
    NumPy C code that releases the GIL, giving true parallelism.
    """
    pool = _get_thread_pool()
    tasks = [(b, oc) for b in range(B) for oc in range(out_c)]
    n_tasks = len(tasks)
    chunk_size = max(1, (n_tasks + NUM_THREADS - 1) // NUM_THREADS)

    def _process_chunk(start: int, end: int) -> None:
        for idx in range(start, min(end, n_tasks)):
            b, oc = tasks[idx]
            g = oc % groups
            c_start = g * c_per_group
            inp_volume = x_pad[b, c_start:c_start + c_per_group]
            kernel_volume = weight[oc]
            dense_out = conv3d_core(inp_volume, kernel_volume)
            out[b, oc] = dense_out[::st, ::sh, ::sw].astype(x_pad.dtype)
            if bias is not None:
                out[b, oc] += bias[oc]

    futures = []
    for i in range(NUM_THREADS):
        s = i * chunk_size
        if s < n_tasks:
            futures.append(pool.submit(_process_chunk, s, s + chunk_size))
    for f in futures:
        f.result()


def _conv3d_temporal_parallel(
    x_pad: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    out: np.ndarray,
    B: int, out_c: int, groups: int, c_per_group: int,
    st: int, sh: int, sw: int,
) -> None:
    """Strategy 2: temporal parallelism inside conv3d_core for depthwise convolutions.

    For depthwise convolutions each conv3d_core call processes a single
    input channel with a 3-D kernel (typically 3x3x3), producing moderate
    per-call work.  Parallelising the *temporal* output positions inside
    each call keeps all 4 U54 cores busy on a single, larger task.
    """
    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group
            inp_volume = x_pad[b, c_start:c_start + c_per_group]
            kernel_volume = weight[oc]
            dense_out = _conv3d_core_threaded(inp_volume, kernel_volume)
            out[b, oc] = dense_out[::st, ::sh, ::sw].astype(x_pad.dtype)
            if bias is not None:
                out[b, oc] += bias[oc]


def _conv3d_core_threaded(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """conv3d_core with temporal output positions distributed across threads.

    Each thread computes an independent slice of temporal positions into a
    local buffer, avoiding any concurrent writes to shared memory.
    """
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)

    C, T_in, H_in, W_in = volume.shape
    _, kT, kH, kW = kernel.shape

    T_out = T_in - kT + 1
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    if T_out < NUM_THREADS:
        return conv3d_core(volume, kernel)

    pool = _get_thread_pool()
    out_volume = np.zeros((T_out, H_out, W_out), dtype=np.float32)

    boundaries = [i * T_out // NUM_THREADS for i in range(NUM_THREADS + 1)]

    def _compute_chunk(t_start: int, t_end: int):
        chunk_len = t_end - t_start
        local_out = np.zeros((chunk_len, H_out, W_out), dtype=np.float32)
        for c in range(C):
            for i, tt in enumerate(range(t_start, t_end)):
                for dt in range(kT):
                    k_2d = kernel[c, dt]
                    if not np.any(k_2d):
                        continue
                    filtered = cv2.filter2D(
                        volume[c, tt + dt],
                        cv2.CV_32F,
                        k_2d,
                        anchor=(0, 0),
                        borderType=cv2.BORDER_CONSTANT,
                    )
                    local_out[i] += filtered[:H_out, :W_out]
        return t_start, local_out

    futures = []
    for i in range(NUM_THREADS):
        t_s, t_e = boundaries[i], boundaries[i + 1]
        if t_s < t_e:
            futures.append(pool.submit(_compute_chunk, t_s, t_e))

    for f in futures:
        t_start, local_out = f.result()
        out_volume[t_start:t_start + local_out.shape[0]] = local_out

    return out_volume


# ===================================================================
# METHOD 4: "native" — C shared library via ctypes
# ===================================================================

def conv3d_forward_native(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """Call the C implementation (libconv3d.so) via ctypes.

    The C library handles zero-padding, pthreads parallelism, and
    internal dispatch to pointwise / depthwise / general fast paths.
    """
    if _c_lib is None:
        raise RuntimeError(
            "C backend not available. Build with:  make -C scratch/ops/conv3d_c"
        )

    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    if bias is not None:
        bias = np.ascontiguousarray(bias, dtype=np.float32)

    B, C_in, T, H, W = x.shape
    C_out, _, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    T_out = (T + 2 * pt - kT) // st + 1
    H_out = (H + 2 * ph - kH) // sh + 1
    W_out = (W + 2 * pw - kW) // sw + 1

    out = np.empty((B, C_out, T_out, H_out, W_out), dtype=np.float32)

    _c_lib.conv3d_forward_c(
        x.ctypes.data_as(_c_float_p),
        weight.ctypes.data_as(_c_float_p),
        bias.ctypes.data_as(_c_float_p) if bias is not None else None,
        out.ctypes.data_as(_c_float_p),
        B, C_in, T, H, W,
        C_out, kT, kH, kW,
        st, sh, sw,
        pt, ph, pw,
        groups,
    )

    return out


# ===================================================================
# Shared low-level kernel (used by "fast" and "threaded")
# ===================================================================

def conv3d_core(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)

    C, T_in, H_in, W_in = volume.shape
    _, kT, kH, kW = kernel.shape

    T_out = T_in - kT + 1
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    out_volume = np.zeros((T_out, H_out, W_out), dtype=np.float32)

    for c in range(C):
        for tt in range(T_out):
            for dt in range(kT):
                k_2d = kernel[c, dt]
                if not np.any(k_2d):
                    continue
                filtered = cv2.filter2D(
                    volume[c, tt + dt],
                    cv2.CV_32F,
                    k_2d,
                    anchor=(0, 0),
                    borderType=cv2.BORDER_CONSTANT,
                )
                out_volume[tt] += filtered[:H_out, :W_out]

    return out_volume
