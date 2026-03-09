"""
3D convolution (no PyTorch). Supports standard and depthwise (groups=in_channels).

Uses OpenCV's cv2.filter2D for accelerated convolution. Fallback: conv3d_forward_slow()
for platforms without OpenCV (e.g. some RISC-V embedded systems).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union
import cv2

def _pad_3d(
    x: np.ndarray,
    pad_t: int, pad_h: int, pad_w: int,
) -> np.ndarray:
    """Zero-pad input on (T, H, W). x shape: (B, C, T, H, W)."""
    B, C, T, H, W = x.shape
    out = np.zeros((B, C, T + 2 * pad_t, H + 2 * pad_h, W + 2 * pad_w), dtype=x.dtype)
    out[:, :, pad_t : pad_t + T, pad_h : pad_h + H, pad_w : pad_w + W] = x
    return out


def conv3d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Union[np.ndarray, None],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    groups: int,
) -> np.ndarray:
    # return conv3d_forward_slow(x, weight, bias, stride, padding, groups)
    return conv3d_forward_fast(x, weight, bias, stride, padding, groups)

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

    # 1. Apply padding in software
    x_pad = _pad_3d(x, pt, ph, pw)
    _, _, Tp, Hp, Wp = x_pad.shape

    # 2. Calculate final strided shapes
    T_out = (Tp - kT) // st + 1
    H_out = (Hp - kH) // sh + 1
    W_out = (Wp - kW) // sw + 1
    out = np.zeros((B, out_c, T_out, H_out, W_out), dtype=x.dtype)

    # 3. Carve the 5D tensor into dense 3D tasks for the FPGA
    for b in range(B):
        for oc in range(out_c):
            g = oc % groups
            c_start = g * c_per_group
            c_end = c_start + c_per_group
            
            # Extract contiguous chunks for the DMA transfer
            inp_volume = x_pad[b, c_start:c_end]
            kernel_volume = weight[oc]
            
            # 4. CALL THE FPGA DRIVER (or software simulation)
            # This returns the dense shape: (Tp - kT + 1, Hp - kH + 1, Wp - kW + 1)
            dense_out = conv3d_core(inp_volume, kernel_volume)
            
            # 5. Apply the stride mathematically using slicing
            strided_out = dense_out[::st, ::sh, ::sw]
            
            # 6. Apply bias and store in the final 5D tensor
            out[b, oc] = strided_out.astype(x.dtype)
            if bias is not None:
                out[b, oc] += bias[oc]

    return out

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
                    anchor=(0,0), 
                    borderType=cv2.BORDER_CONSTANT
                )
                out_volume[tt] += filtered[:H_out, :W_out]
                
    return out_volume