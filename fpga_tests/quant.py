"""
Symmetric int8 quantization primitives used by the FPGA bring-up harness.

Everything here matches section 4 of fpga_flow.md:
  * Signed 8-bit, clipped to [-127, +127] (we deliberately drop -128).
  * Per-tensor scales for activations, per-output-channel scales for weights.
  * Requantization multiplier M[c] = s_in * s_w[c] / s_out.
  * Fixed-point form M[c] ≈ M0[c] * 2^(-n[c]) for the "FPGA" path.

These helpers are deliberately tiny and dependency-free (just NumPy). The
goal is that every single step can be inspected by eye for a small test
tensor — there is no magic.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

INT8_MAX = 127
INT8_MIN = -127


def compute_tensor_scale(x: np.ndarray) -> np.float32:
    """Per-tensor symmetric scale: s = max(|x|) / 127. Guards against all-zero."""
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    if amax == 0.0:
        amax = 1.0
    return np.float32(amax / INT8_MAX)


def compute_weight_scales(W: np.ndarray) -> np.ndarray:
    """
    Per-output-channel weight scales. W is assumed (O, I/g, kT, kH, kW).
    Returns float32 vector of shape (O,).
    """
    O = W.shape[0]
    flat = W.reshape(O, -1)
    amax = np.max(np.abs(flat), axis=1)
    amax[amax == 0.0] = 1.0
    return (amax / INT8_MAX).astype(np.float32)


def quantize_tensor(x: np.ndarray, s: float) -> np.ndarray:
    """float32 tensor -> int8, symmetric, clipped."""
    q = np.round(x / s)
    q = np.clip(q, INT8_MIN, INT8_MAX)
    return q.astype(np.int8)


def quantize_weights(W: np.ndarray, s_w: np.ndarray) -> np.ndarray:
    """
    Per-channel weight quantization. W: (O, I/g, kT, kH, kW). s_w: (O,).
    Returns int8 tensor of the same shape as W.
    """
    s = s_w.reshape(-1, 1, 1, 1, 1).astype(np.float32)
    q = np.round(W / s)
    q = np.clip(q, INT8_MIN, INT8_MAX)
    return q.astype(np.int8)


def dequantize_tensor(q: np.ndarray, s: float) -> np.ndarray:
    """int8 tensor -> float32."""
    return q.astype(np.float32) * np.float32(s)


# ---------------------------------------------------------------------------
# Requantization (§5 of fpga_flow.md)
# ---------------------------------------------------------------------------

def compute_M(s_in: float, s_w: np.ndarray, s_out: float) -> np.ndarray:
    """M[c] = (s_in * s_w[c]) / s_out — per-channel float32 multiplier."""
    return ((np.float32(s_in) * s_w.astype(np.float32)) / np.float32(s_out)).astype(np.float32)


def quantize_multiplier_fixed_point(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Represent each float32 multiplier M[c] (expected to be in (0, 1)) as
        M[c] ≈ M0[c] * 2^(-n[c])
    where M0[c] is an int32 in [2^30, 2^31 - 1] and n[c] is a non-negative int.

    This is the same decomposition TFLite / CMSIS-NN use. The CPU computes
    (M0, n) once at model-build time; the FPGA only ever sees the int pair.
    """
    M = M.astype(np.float64)
    O = M.shape[0]
    M0 = np.zeros(O, dtype=np.int64)
    n = np.zeros(O, dtype=np.int32)
    for c in range(O):
        m = float(M[c])
        if m <= 0.0:
            M0[c] = 0
            n[c] = 0
            continue
        # Normalize mantissa into [0.5, 1.0), track the exponent.
        mantissa, exp = np.frexp(m)       # m = mantissa * 2^exp, mantissa in [0.5, 1)
        # Scale mantissa into int32 range [2^30, 2^31 - 1].
        q = int(round(mantissa * (1 << 31)))
        if q == (1 << 31):                # overflow after rounding
            q //= 2
            exp += 1
        # m ≈ q * 2^(exp - 31), so n = 31 - exp (must be >= 0 for m in (0,1))
        shift = 31 - exp
        if shift < 0:
            # M >= 1; clamp. Should not happen for well-calibrated models.
            q = (1 << 31) - 1
            shift = 0
        M0[c] = q
        n[c] = shift
    return M0.astype(np.int64), n.astype(np.int32)


def apply_requantize_float(acc32: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Software-reference requantization: float32 multiply.
    acc32: (B, O, T, H, W) int32. M: (O,) float32.
    Returns int8 of the same shape.
    """
    M_b = M.reshape(1, -1, 1, 1, 1).astype(np.float32)
    y = np.round(acc32.astype(np.float32) * M_b)
    y = np.clip(y, INT8_MIN, INT8_MAX)
    return y.astype(np.int8)


def apply_requantize_fixed_point(
    acc32: np.ndarray, M0: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    "FPGA" requantization: int64 multiply + rounding right shift.

      y[c] = saturate_int8( round_nearest( (acc32[c] * M0[c]) >> n[c] ) )

    The rounding rule is round-half-away-from-zero, implemented by adding
    (1 << (n-1)) before the shift (for nonnegative values) and the symmetric
    form for negatives. This matches the usual TFLite/CMSIS-NN convention
    closely enough for bring-up; a real FPGA will pick one canonical rounding
    mode and stick to it.
    """
    assert acc32.dtype == np.int32
    B, O, T, H, W = acc32.shape
    M0_b = M0.reshape(1, -1, 1, 1, 1).astype(np.int64)
    n_b = n.reshape(1, -1, 1, 1, 1).astype(np.int64)

    prod = acc32.astype(np.int64) * M0_b  # int64 mul
    # Rounding offset: +/- 2^(n-1), zero when n == 0.
    half = np.where(n_b > 0, np.int64(1) << np.maximum(n_b - 1, 0), np.int64(0))
    sign = np.where(prod >= 0, np.int64(1), np.int64(-1))
    rounded = (prod + sign * half) >> n_b
    rounded = np.clip(rounded, INT8_MIN, INT8_MAX)
    return rounded.astype(np.int8)
