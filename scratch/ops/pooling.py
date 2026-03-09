"""
3D pooling (no PyTorch): AvgPool3d and AdaptiveAvgPool3d.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union


def avg_pool3d_forward(
    x: np.ndarray,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[None, int, Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Average pooling over 3D windows. Layout: (B, C, T, H, W).

    Args:
        x: Input (B, C, T, H, W).
        kernel_size: (kT, kH, kW) or single int for all.
        stride: Step size; if None, equals kernel_size.

    Returns:
        Pooled tensor (B, C, T', H', W').
    """
    if isinstance(kernel_size, int):
        kT = kH = kW = kernel_size
    else:
        kT, kH, kW = kernel_size
    if stride is None:
        st, sh, sw = kT, kH, kW
    elif isinstance(stride, int):
        st = sh = sw = stride
    else:
        st, sh, sw = stride

    B, C, T, H, W = x.shape
    T_out = (T - kT) // st + 1
    H_out = (H - kH) // sh + 1
    W_out = (W - kW) // sw + 1
    out = np.zeros((B, C, T_out, H_out, W_out), dtype=x.dtype)
    n = kT * kH * kW

    for tt in range(T_out):
        for hh in range(H_out):
            for ww in range(W_out):
                t0, h0, w0 = tt * st, hh * sh, ww * sw
                out[:, :, tt, hh, ww] = np.mean(
                    x[:, :, t0 : t0 + kT, h0 : h0 + kH, w0 : w0 + kW],
                    axis=(2, 3, 4),
                )
    return out


def adaptive_avg_pool3d_forward(
    x: np.ndarray,
    output_size: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Adaptive average pooling to fixed output size. Layout: (B, C, T, H, W).

    Args:
        x: Input (B, C, T, H, W).
        output_size: (oT, oH, oW) or single int for (1,1,1) (global pool).

    Returns:
        Pooled tensor (B, C, oT, oH, oW).
    """
    if isinstance(output_size, int):
        oT = oH = oW = output_size
    else:
        oT, oH, oW = output_size

    B, C, T, H, W = x.shape
    out = np.zeros((B, C, oT, oH, oW), dtype=x.dtype)

    for tt in range(oT):
        for hh in range(oH):
            for ww in range(oW):
                t_start = (tt * T) // oT
                t_end = ((tt + 1) * T) // oT
                h_start = (hh * H) // oH
                h_end = ((hh + 1) * H) // oH
                w_start = (ww * W) // oW
                w_end = ((ww + 1) * W) // oW
                out[:, :, tt, hh, ww] = np.mean(
                    x[:, :, t_start:t_end, h_start:h_end, w_start:w_end],
                    axis=(2, 3, 4),
                )
    return out
