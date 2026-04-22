"""
Int8 3D convolution (no PyTorch). Supports standard and depthwise (groups=in_channels).

Mirrors conv3d.py but operates on quantized int8 tensors:
  int8 input x int8 weights -> int32 accumulate -> int32 output

The raw int32 accumulator is returned directly — no requantization is performed.
This is useful when requantization is handled externally (e.g. by the caller or
a downstream layer), or when the int32 output is needed for analysis/debugging.

Four implementation strategies selectable via set_conv3d_int8_method() or per-call:

  "slow"     — Pure NumPy fallback. 6-deep nested loops with int32 accumulation.
               Correctness reference only.

  "fast"     — NumPy-accelerated path. Uses manual spatial loops with vectorised
               channel accumulation. Loops over temporal/spatial dimensions.

  "threaded" — Multi-threaded NumPy path targeting the PolarFire SoC's 4 U54
               RISC-V application cores:
                 - Pointwise / standard convolutions: output-channel parallelism
                 - Depthwise convolutions: temporal parallelism
               NumPy releases the GIL during C-level computation, so Python
               threads achieve genuine parallelism for these workloads.

  "native"   — C shared-library backend (libconv3d_int8.so) loaded via ctypes.
               Pthreads parallelism (4 threads), cache-friendly spatial tiling,
               separate fast paths for pointwise / depthwise / general conv.
               Build with: make -C scratch/ops/conv3d_c_int8
               Falls back to RuntimeError if the library has not been compiled.

Default method: "fast"

Note: OpenCV's cv2.filter2D only supports float/uint8 output — not signed int8
with int32 accumulation. The fast/threaded methods here use pure NumPy instead.
"""

from __future__ import annotations
import ctypes
import os
import numpy as np
from typing import Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

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
_c_int8_p = ctypes.POINTER(ctypes.c_int8)
_c_int32_p = ctypes.POINTER(ctypes.c_int32)


def _load_c_backend() -> None:
    global _c_lib
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv3d_c_int8")

    for name in ("libconv3d_int8.so", "libconv3d_int8.dylib"):
        lib_path = os.path.join(lib_dir, name)
        if os.path.isfile(lib_path):
            break
    else:
        print("[conv3d_int8] C backend not found — build with:  "
              "make -C scratch/ops/conv3d_c_int8")
        return

    try:
        _c_lib = ctypes.CDLL(lib_path)
        _c_lib.conv3d_int8_forward_c.restype = None
        _c_lib.conv3d_int8_forward_c.argtypes = [
            _c_int8_p,                                      # input
            _c_int8_p,                                      # weight
            _c_int32_p,                                     # output (int32)
            ctypes.c_int, ctypes.c_int,                     # B, C_in
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # T, H, W
            ctypes.c_int,                                   # C_out
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # kT, kH, kW
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # stride_t/h/w
            ctypes.c_int, ctypes.c_int, ctypes.c_int,       # pad_t/h/w
            ctypes.c_int,                                   # groups
        ]
        print(f"[conv3d_int8] C backend loaded successfully from {lib_path}")
    except OSError as e:
        print(f"[conv3d_int8] C backend load failed: {e}")


_load_c_backend()


def is_native_available() -> bool:
    """Return True if the int8 C shared-library backend is loaded."""
    return _c_lib is not None


def _get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)
    return _thread_pool


def set_conv3d_int8_method(method: str) -> None:
    """Set the default int8 convolution implementation.

    Args:
        method: One of "slow", "fast", "threaded", or "native".
    """
    global _default_method
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}, got {method!r}"
        )
    _default_method = method


def get_conv3d_int8_method() -> str:
    """Return the current default int8 convolution method."""
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

def _pad_3d_int8(
    x: np.ndarray,
    pad_t: int, pad_h: int, pad_w: int,
) -> np.ndarray:
    """Zero-pad int8 input on (T, H, W). x shape: (B, C, T, H, W)."""
    B, C, T, H, W = x.shape
    out = np.zeros(
        (B, C, T + 2 * pad_t, H + 2 * pad_h, W + 2 * pad_w), dtype=np.int8
    )
    out[:, :, pad_t : pad_t + T, pad_h : pad_h + H, pad_w : pad_w + W] = x
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def conv3d_int8_forward(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
    method: Optional[str] = None,
) -> np.ndarray:
    """Dispatch to the selected int8 3D convolution implementation.

    Args:
        x:       Input  (B, C_in, T, H, W) int8.
        weight:  Kernel (C_out, C_in//groups, kT, kH, kW) int8.
        stride:  (stride_t, stride_h, stride_w).
        padding: (pad_t, pad_h, pad_w).
        groups:  1 = standard; C_in = depthwise.
        method:  Override the global default ("slow", "fast", "threaded", "native").
                 If None, uses the value set by set_conv3d_int8_method().

    Returns:
        Output tensor (B, C_out, T_out, H_out, W_out) int32.
    """
    if method is None:
        method = _default_method
    if method == "slow":
        return conv3d_int8_forward_slow(x, weight, stride, padding, groups)
    if method == "fast":
        return conv3d_int8_forward_fast(x, weight, stride, padding, groups)
    if method == "threaded":
        return conv3d_int8_forward_threaded(x, weight, stride, padding, groups)
    if method == "native":
        return conv3d_int8_forward_native(x, weight, stride, padding, groups)
    raise ValueError(
        f"Unknown conv3d_int8 method {method!r}; choose from {VALID_METHODS}"
    )


# ===================================================================
# METHOD 1: "slow" — pure NumPy (correctness reference)
# ===================================================================

def conv3d_int8_forward_slow(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """
    Int8 3D convolution — pure NumPy, 6-deep nested loops.

    int8 x int8 -> int32 accumulate -> int32 output.

    Args:
        x:      Input  (B, C_in, T, H, W) int8.
        weight: Kernel (C_out, C_in//groups, kT, kH, kW) int8.
        stride: (stride_t, stride_h, stride_w).
        padding: (pad_t, pad_h, pad_w).
        groups: Number of groups.

    Returns:
        Output tensor (B, C_out, T_out, H_out, W_out) int32.
    """
    B, in_c, T, H, W = x.shape
    out_c, c_per_group, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    if in_c % groups != 0 or out_c % groups != 0:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if in_c // groups != c_per_group:
        raise ValueError("weight in_channels per group mismatch")

    x_pad = _pad_3d_int8(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1

    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=np.int32)

    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group

            for tt in range(T_out):
                for hh in range(H_out):
                    for ww in range(W_out):
                        acc = np.int32(0)
                        t0, h0, w0 = tt * st, hh * sh, ww * sw
                        for c in range(c_per_group):
                            w_slice = weight[oc, c, :, :, :]  # (kT, kH, kW)
                            x_slice = x_pad[b, c_start + c,
                                            t0 : t0 + kT,
                                            h0 : h0 + kH,
                                            w0 : w0 + kW]
                            # int8 x int8 -> int32 accumulation
                            acc += np.int32(np.sum(
                                x_slice.astype(np.int32) * w_slice.astype(np.int32)
                            ))

                        out[b, oc, tt, hh, ww] = acc

    return out


# ===================================================================
# METHOD 2: "fast" — vectorised NumPy (single-threaded)
# ===================================================================

def conv3d_int8_forward_fast(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """
    Int8 3D convolution — vectorised channel accumulation with NumPy.

    Uses int32 dot products over the channel+kernel dimensions.
    Returns raw int32 accumulator output.
    """
    B, in_c, T, H, W = x.shape
    out_c, c_per_group, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    x_pad = _pad_3d_int8(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1
    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=np.int32)

    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group

            # Get the kernel for this output channel: (c_per_group, kT, kH, kW)
            kern = weight[oc].astype(np.int32)  # (cpg, kT, kH, kW)

            acc_volume = conv3d_core_int8(
                x_pad[b, c_start:c_start + c_per_group], kern
            )

            # Apply stride
            out[b, oc] = acc_volume[::st, ::sh, ::sw]  # (T_out, H_out, W_out) int32

    return out


# ===================================================================
# METHOD 3: "threaded" — multi-threaded with adaptive strategy
# ===================================================================

def conv3d_int8_forward_threaded(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """Multi-threaded int8 conv3d with adaptive hybrid parallelism.

    Pointwise / standard convolutions use output-channel parallelism
    (Strategy 1).  Depthwise convolutions use temporal parallelism
    within conv3d_core_int8 (Strategy 2).
    """
    B, in_c, T, H, W = x.shape
    out_c, c_per_group, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    x_pad = _pad_3d_int8(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1
    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=np.int32)

    is_pointwise = kT * kH * kW == 1
    is_depthwise = groups == out_c and groups > 1

    if is_pointwise or not is_depthwise:
        _conv3d_int8_oc_parallel(x_pad, weight, out,
                                  B, out_c, groups, c_per_group, st, sh, sw)
    else:
        _conv3d_int8_temporal_parallel(x_pad, weight, out,
                                        B, out_c, groups, c_per_group, st, sh, sw)

    return out


def _conv3d_int8_oc_parallel(
    x_pad: np.ndarray,
    weight: np.ndarray,
    out: np.ndarray,
    B: int, out_c: int, groups: int, c_per_group: int,
    st: int, sh: int, sw: int,
) -> None:
    """Strategy 1: distribute (batch, output-channel) pairs across threads."""
    pool = _get_thread_pool()
    tasks = [(b, oc) for b in range(B) for oc in range(out_c)]
    n_tasks = len(tasks)
    chunk_size = max(1, (n_tasks + NUM_THREADS - 1) // NUM_THREADS)

    def _process_chunk(start: int, end: int) -> None:
        for idx in range(start, min(end, n_tasks)):
            b, oc = tasks[idx]
            g = oc % groups
            c_start = g * c_per_group

            kern = weight[oc].astype(np.int32)
            acc_volume = conv3d_core_int8(
                x_pad[b, c_start:c_start + c_per_group], kern
            )
            out[b, oc] = acc_volume[::st, ::sh, ::sw]

    futures = []
    for i in range(NUM_THREADS):
        s = i * chunk_size
        if s < n_tasks:
            futures.append(pool.submit(_process_chunk, s, s + chunk_size))
    for f in futures:
        f.result()


def _conv3d_int8_temporal_parallel(
    x_pad: np.ndarray,
    weight: np.ndarray,
    out: np.ndarray,
    B: int, out_c: int, groups: int, c_per_group: int,
    st: int, sh: int, sw: int,
) -> None:
    """Strategy 2: temporal parallelism inside conv3d_core for depthwise convolutions."""
    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group

            kern = weight[oc].astype(np.int32)
            acc_volume = _conv3d_core_int8_threaded(
                x_pad[b, c_start:c_start + c_per_group], kern
            )
            out[b, oc] = acc_volume[::st, ::sh, ::sw]


def _conv3d_core_int8_threaded(
    volume: np.ndarray, kernel: np.ndarray,
) -> np.ndarray:
    """conv3d_core_int8 with temporal output positions distributed across threads.

    Each thread computes an independent slice of temporal positions into a
    local buffer, avoiding any concurrent writes to shared memory.
    """
    volume = np.ascontiguousarray(volume, dtype=np.int8)
    kernel = np.ascontiguousarray(kernel, dtype=np.int32)

    C, T_in, H_in, W_in = volume.shape
    _, kT, kH, kW = kernel.shape

    T_out = T_in - kT + 1
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    if T_out < NUM_THREADS:
        return conv3d_core_int8(volume, kernel)

    pool = _get_thread_pool()
    out_volume = np.zeros((T_out, H_out, W_out), dtype=np.int32)

    boundaries = [i * T_out // NUM_THREADS for i in range(NUM_THREADS + 1)]

    def _compute_chunk(t_start: int, t_end: int):
        chunk_len = t_end - t_start
        local_out = np.zeros((chunk_len, H_out, W_out), dtype=np.int32)
        for c in range(C):
            for i, tt in enumerate(range(t_start, t_end)):
                for dt in range(kT):
                    k_2d = kernel[c, dt]  # (kH, kW) int32
                    if not np.any(k_2d):
                        continue
                    # Extract the 2D input slice and compute int32 correlation
                    inp_2d = volume[c, tt + dt].astype(np.int32)  # (H_in, W_in)
                    # Manual 2D correlation (no cv2.filter2D for int32)
                    for dh in range(kH):
                        for dw in range(kW):
                            local_out[i] += k_2d[dh, dw] * inp_2d[dh:dh + H_out, dw:dw + W_out]
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

def conv3d_int8_forward_native(
    x: np.ndarray,
    weight: np.ndarray,
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    """Call the int8 C implementation (libconv3d_int8.so) via ctypes.

    The C library handles zero-padding, pthreads parallelism, and
    internal dispatch to pointwise / depthwise / general fast paths.
    Returns the raw int32 accumulator output.
    """
    if _c_lib is None:
        raise RuntimeError(
            "Int8 C backend not available. Build with:  "
            "make -C scratch/ops/conv3d_c_int8"
        )

    x = np.ascontiguousarray(x, dtype=np.int8)
    weight = np.ascontiguousarray(weight, dtype=np.int8)

    B, C_in, T, H, W = x.shape
    C_out, _, kT, kH, kW = weight.shape
    st, sh, sw = stride
    pt, ph, pw = padding

    T_out = (T + 2 * pt - kT) // st + 1
    H_out = (H + 2 * ph - kH) // sh + 1
    W_out = (W + 2 * pw - kW) // sw + 1

    out = np.empty((B, C_out, T_out, H_out, W_out), dtype=np.int32)

    _c_lib.conv3d_int8_forward_c(
        x.ctypes.data_as(_c_int8_p),
        weight.ctypes.data_as(_c_int8_p),
        out.ctypes.data_as(_c_int32_p),
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

def conv3d_core_int8(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute dense 3D correlation with int32 accumulation.

    Unlike the float32 conv3d_core which uses cv2.filter2D, this version
    uses pure NumPy because cv2.filter2D does not support int32 accumulation
    of int8 inputs natively.

    Args:
        volume: (C, T_in, H_in, W_in) int8 — already padded.
        kernel: (C, kT, kH, kW) int32 — weights pre-cast to int32.

    Returns:
        (T_out, H_out, W_out) int32 — raw accumulator.
    """
    volume = np.ascontiguousarray(volume, dtype=np.int8)
    kernel = np.ascontiguousarray(kernel, dtype=np.int32)

    C, T_in, H_in, W_in = volume.shape
    _, kT, kH, kW = kernel.shape

    T_out = T_in - kT + 1
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    out_volume = np.zeros((T_out, H_out, W_out), dtype=np.int32)

    for c in range(C):
        for tt in range(T_out):
            for dt in range(kT):
                k_2d = kernel[c, dt]  # (kH, kW) int32
                if not np.any(k_2d):
                    continue
                # 2D correlation via vectorised slicing
                inp_2d = volume[c, tt + dt].astype(np.int32)  # (H_in, W_in)
                for dh in range(kH):
                    for dw in range(kW):
                        out_volume[tt] += k_2d[dh, dw] * inp_2d[dh:dh + H_out, dw:dw + W_out]

    return out_volume
