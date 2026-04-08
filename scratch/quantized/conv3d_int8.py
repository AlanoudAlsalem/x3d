"""
Software reference int8 3D convolution kernel.

This file implements, in pure NumPy, the exact sequence of operations that
the future FPGA accelerator will perform for one quantized Conv3d layer:

    1. int8 * int8 multiplication accumulated into int32
    2. int32 bias addition
    3. requantization to int8 using a per-output-channel multiplier M[c]
       derived from the input, weight, and output scales

This is NOT meant to be fast. It is meant to be a bit-accurate model of the
hardware so that you can develop, debug, and validate the rest of the
quantized pipeline without waiting for the FPGA bitstream. When the FPGA is
ready, this function's outputs should match the FPGA outputs exactly (once
the fixed-point requantization is wired in; today it uses float32 for M).

Input/output layout matches the float32 kernel: (B, C, T, H, W).
Weights: (out_channels, in_channels // groups, kT, kH, kW) as int8.
Bias: (out_channels,) as int32 or None.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _pad_3d_int8(x: np.ndarray, padding: Tuple[int, int, int]) -> np.ndarray:
    """
    Zero-pad an int8 tensor along (T, H, W) only. Batch and channel dims
    are not padded.

    Zero is the representative of the real value 0.0 under symmetric
    quantization, so padding with integer 0 is semantically the same as
    padding the underlying float tensor with 0.0.
    """
    pT, pH, pW = padding
    if pT == 0 and pH == 0 and pW == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (pT, pT), (pH, pH), (pW, pW)),
        mode="constant",
        constant_values=0,
    )


def conv3d_int8_forward(
    x_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: Optional[np.ndarray],
    input_scale: float,
    weight_scale: np.ndarray,
    output_scale: float,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    groups: int = 1,
) -> np.ndarray:
    """
    Reference int8 3D convolution (software proxy for the FPGA).

    Parameters
    ----------
    x_q : np.ndarray, int8, shape (B, C_in, T, H, W)
        Quantized input activation. Must already be int8 with an implicit
        scale of ``input_scale``.
    weight_q : np.ndarray, int8, shape (C_out, C_in // groups, kT, kH, kW)
        Quantized weights with an implicit per-channel scale of ``weight_scale``.
    bias_q : np.ndarray or None, int32, shape (C_out,)
        Quantized bias in int32 accumulator units. ``bias_q[c]`` has an
        implicit scale of ``input_scale * weight_scale[c]`` and therefore
        can be added directly into the int32 accumulator before
        requantization. Pass None if the layer has no bias.
    input_scale : float
        Scalar float32 scale of the input activation tensor.
    weight_scale : np.ndarray, float32, shape (C_out,)
        Per-output-channel weight scale.
    output_scale : float
        Scalar float32 scale of the output activation tensor.
    stride : (sT, sH, sW)
    padding : (pT, pH, pW)
    groups : int
        1 for standard conv, C_in for depthwise.

    Returns
    -------
    y_q : np.ndarray, int8, shape (B, C_out, T', H', W')
        Quantized output activation, ready to be dequantized with
        ``y_q.astype(np.float32) * output_scale``.

    Notes
    -----
    This implementation is intentionally simple: nested loops over spatial
    positions, dispatching one 2D convolution (or rather, 3D patch multiply)
    per position. It is O(slow) but correct. Speed does not matter for this
    kernel because its only job is to validate the pipeline before the real
    FPGA runs the same math in hardware.

    The per-channel requantization multiplier is computed as
    ``M[c] = (input_scale * weight_scale[c]) / output_scale`` and applied in
    float32 for simplicity. A future iteration will replace this with the
    fixed-point ``(M0[c], n[c])`` form described in fpga_flow.md §5.2.
    """
    assert x_q.dtype == np.int8, f"x_q must be int8, got {x_q.dtype}"
    assert weight_q.dtype == np.int8, f"weight_q must be int8, got {weight_q.dtype}"
    if bias_q is not None:
        assert bias_q.dtype == np.int32, f"bias_q must be int32, got {bias_q.dtype}"

    B, C_in, T, H, W = x_q.shape
    C_out, C_in_per_group, kT, kH, kW = weight_q.shape
    sT, sH, sW = stride
    pT, pH, pW = padding

    assert C_in % groups == 0, "in_channels must be divisible by groups"
    assert C_out % groups == 0, "out_channels must be divisible by groups"
    assert C_in // groups == C_in_per_group, (
        f"weight in-channels {C_in_per_group} does not match "
        f"in_channels/groups = {C_in // groups}"
    )

    # Pad input (padding with 0 is exact under symmetric quantization).
    x_padded = _pad_3d_int8(x_q, padding)
    _, _, Tp, Hp, Wp = x_padded.shape

    # Output spatial shape.
    T_out = (Tp - kT) // sT + 1
    H_out = (Hp - kH) // sH + 1
    W_out = (Wp - kW) // sW + 1

    # Accumulator lives in int32. We compute in int32 throughout.
    # Promote once here; all subsequent math stays int32 until requant.
    x_i32 = x_padded.astype(np.int32)
    w_i32 = weight_q.astype(np.int32)

    acc = np.zeros((B, C_out, T_out, H_out, W_out), dtype=np.int32)

    C_out_per_group = C_out // groups

    # Main triple loop over spatial output positions. Hoisting (t, h, w) out
    # of the channel loops lets us use a single slicing expression per step.
    for t in range(T_out):
        t0 = t * sT
        for h in range(H_out):
            h0 = h * sH
            for w in range(W_out):
                w0 = w * sW
                # Patch: (B, C_in, kT, kH, kW)
                patch = x_i32[:, :, t0:t0 + kT, h0:h0 + kH, w0:w0 + kW]

                for g in range(groups):
                    c_in_start = g * C_in_per_group
                    c_in_end = c_in_start + C_in_per_group
                    c_out_start = g * C_out_per_group
                    c_out_end = c_out_start + C_out_per_group

                    # (B, C_in_per_group, kT, kH, kW) @ (C_out_per_group, C_in_per_group, kT, kH, kW)
                    # -> (B, C_out_per_group)
                    patch_g = patch[:, c_in_start:c_in_end, :, :, :]
                    w_g = w_i32[c_out_start:c_out_end, :, :, :, :]

                    # Flatten the per-output-channel patch and weight so we
                    # can use a single tensordot. This performs the int32
                    # accumulation natively.
                    patch_flat = patch_g.reshape(B, -1)                # (B, K)
                    w_flat = w_g.reshape(C_out_per_group, -1)           # (C_out_per_group, K)

                    acc[:, c_out_start:c_out_end, t, h, w] = patch_flat @ w_flat.T

    # Add int32 bias directly into the accumulator.
    if bias_q is not None:
        acc += bias_q.reshape(1, C_out, 1, 1, 1)

    # Requantize to int8 using per-channel M[c] = s_in * s_w[c] / s_out.
    # Done in float32 here for simplicity. Hardware will use fixed-point.
    M = (input_scale * weight_scale.astype(np.float32)) / np.float32(output_scale)
    M = M.reshape(1, C_out, 1, 1, 1)
    y_f = acc.astype(np.float32) * M
    # Round to nearest, ties to even (numpy default), then clip and cast.
    y_rounded = np.rint(y_f)
    np.clip(y_rounded, -127, 127, out=y_rounded)
    y_q = y_rounded.astype(np.int8)

    return y_q
