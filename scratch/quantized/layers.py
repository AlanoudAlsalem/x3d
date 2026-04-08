"""
Quantized layer modules for the int8 X3D-M runtime.

QuantizedConv3d replaces scratch.nn.conv3d.Conv3d in the int8 model. It holds
int8 weights, an int32 bias, per-channel weight scales, and per-tensor input
and output activation scales. Its forward() method performs the
quantize -> (int8 conv) -> dequantize sequence described in fpga_flow.md §7.

QuantizedLinear does the same for the classification head.

Both modules expose a ``backend`` attribute controlling where the int8 conv
actually runs:

    "reference"  — pure NumPy software kernel in conv3d_int8.py. This is the
                   default and is what you use for pipeline bring-up.
    "fpga"       — placeholder for the real FPGA driver. Raises
                   NotImplementedError until you wire it up to your hardware.

The float32 path (scratch.nn.conv3d.Conv3d) is untouched; these modules live
in their own subpackage and are only instantiated by the quantized model.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from scratch.nn.module import Module
from scratch.quantized.conv3d_int8 import conv3d_int8_forward


def _triple(v: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    return tuple(v)  # type: ignore[return-value]


class QuantizedConv3d(Module):
    """
    Int8 3D convolution layer with float32 boundary quantize/dequantize.

    Parameters match scratch.nn.conv3d.Conv3d so this layer can be swapped in
    structurally. Internally, however, it stores int8 weights and a set of
    scale factors instead of a float32 weight.

    Storage (in self._parameters):

        weight_q      : int8,    (C_out, C_in // groups, kT, kH, kW)
        weight_scale  : float32, (C_out,)
        bias_q        : int32,   (C_out,)    or None
        input_scale   : float32, shape ()
        output_scale  : float32, shape ()

    Forward pass:

        1. Quantize the incoming float32 activation using input_scale.
        2. Run the int8 conv kernel (reference or FPGA backend).
        3. Dequantize the int8 result using output_scale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        bias: bool = True,
        groups: int = 1,
        backend: str = "reference",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.groups = groups
        self.backend = backend

        kT, kH, kW = self.kernel_size
        c_per_group = in_channels // groups

        # Zero-initialize; real values must be loaded from an int8 .npz.
        self._parameters["weight_q"] = np.zeros(
            (out_channels, c_per_group, kT, kH, kW), dtype=np.int8
        )
        self._parameters["weight_scale"] = np.ones((out_channels,), dtype=np.float32)
        if bias:
            self._parameters["bias_q"] = np.zeros((out_channels,), dtype=np.int32)
        else:
            self._parameters["bias_q"] = None
        # Scalars are stored as 0-d float32 arrays so the Module param walker
        # can treat them uniformly with the other parameters.
        self._parameters["input_scale"] = np.float32(1.0)
        self._parameters["output_scale"] = np.float32(1.0)

    # ---- Convenience accessors -------------------------------------------------
    @property
    def weight_q(self) -> np.ndarray:
        return self._parameters["weight_q"]

    @property
    def weight_scale(self) -> np.ndarray:
        return self._parameters["weight_scale"]

    @property
    def bias_q(self) -> Optional[np.ndarray]:
        return self._parameters["bias_q"]

    @property
    def input_scale(self) -> float:
        return float(self._parameters["input_scale"])

    @property
    def output_scale(self) -> float:
        return float(self._parameters["output_scale"])

    # ---- Core ------------------------------------------------------------------

    def _quantize_input(self, x_f32: np.ndarray) -> np.ndarray:
        """
        CPU-side: float32 -> int8 using per-tensor symmetric scale.

        This is step (1) in fpga_flow.md §7.2. On the SoC this could be
        replaced with a small C kernel for speed; here we use NumPy.
        """
        s = self.input_scale
        if s == 0.0:
            # Degenerate layer (never seen a non-zero activation during
            # calibration). Return zeros rather than NaN.
            return np.zeros_like(x_f32, dtype=np.int8)
        q = np.rint(x_f32 / s)
        np.clip(q, -127, 127, out=q)
        return q.astype(np.int8)

    def _dequantize_output(self, y_q: np.ndarray) -> np.ndarray:
        """
        CPU-side: int8 -> float32 using per-tensor symmetric scale.

        This is step (7) in fpga_flow.md §7.2.
        """
        return y_q.astype(np.float32) * np.float32(self.output_scale)

    def _run_backend(self, x_q: np.ndarray) -> np.ndarray:
        """Dispatch the int8 convolution to the configured backend."""
        if self.backend == "reference":
            return conv3d_int8_forward(
                x_q=x_q,
                weight_q=self.weight_q,
                bias_q=self.bias_q,
                input_scale=self.input_scale,
                weight_scale=self.weight_scale,
                output_scale=self.output_scale,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )
        if self.backend == "fpga":
            raise NotImplementedError(
                "FPGA backend not wired up yet. Implement a driver that "
                "hands (x_q, weight_q, bias_q, M_table) to the accelerator "
                "and returns the int8 output."
            )
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        float32 in -> float32 out, with int8 conv in the middle.

        The three-step dance here is exactly the CPU/FPGA handshake
        described in fpga_flow.md §7.2, simply executed in Python:

            x_f32 --quantize--> x_q --conv_int8--> y_q --dequantize--> y_f32
        """
        assert x.dtype == np.float32, f"QuantizedConv3d expects float32 input, got {x.dtype}"
        x_q = self._quantize_input(x)
        y_q = self._run_backend(x_q)
        return self._dequantize_output(y_q)


class QuantizedLinear(Module):
    """
    Int8 fully connected layer used for the X3D-M classification head.

    This mirrors the logic of QuantizedConv3d but for a 2D weight matrix.
    Because there is only one Linear in X3D-M (the final 2048 -> 400 logits
    projection), it is kept simple and uses an explicit NumPy matmul in
    int32, exactly analogous to the conv kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._parameters["weight_q"] = np.zeros((out_features, in_features), dtype=np.int8)
        self._parameters["weight_scale"] = np.ones((out_features,), dtype=np.float32)
        if bias:
            self._parameters["bias_q"] = np.zeros((out_features,), dtype=np.int32)
        else:
            self._parameters["bias_q"] = None
        self._parameters["input_scale"] = np.float32(1.0)
        self._parameters["output_scale"] = np.float32(1.0)

    @property
    def weight_q(self) -> np.ndarray:
        return self._parameters["weight_q"]

    @property
    def weight_scale(self) -> np.ndarray:
        return self._parameters["weight_scale"]

    @property
    def bias_q(self) -> Optional[np.ndarray]:
        return self._parameters["bias_q"]

    @property
    def input_scale(self) -> float:
        return float(self._parameters["input_scale"])

    @property
    def output_scale(self) -> float:
        return float(self._parameters["output_scale"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        float32 in, float32 out. (B, in_features) -> (B, out_features).

        The X3D-M head flattens to (B, in_features) before the Linear, so we
        expect a 2D input here. If a higher-rank tensor arrives we flatten
        the trailing dims as PyTorch's Linear does.
        """
        assert x.dtype == np.float32

        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(orig_shape[0], -1)

        s_in = self.input_scale
        if s_in == 0.0:
            x_q = np.zeros_like(x, dtype=np.int8)
        else:
            q = np.rint(x / s_in)
            np.clip(q, -127, 127, out=q)
            x_q = q.astype(np.int8)

        # (B, in_features) @ (in_features, out_features) in int32
        acc = x_q.astype(np.int32) @ self.weight_q.astype(np.int32).T   # (B, out_features)
        if self.bias_q is not None:
            acc = acc + self.bias_q.reshape(1, -1)

        M = (s_in * self.weight_scale.astype(np.float32)) / np.float32(self.output_scale)
        y_f = acc.astype(np.float32) * M.reshape(1, -1)
        y_rounded = np.rint(y_f)
        np.clip(y_rounded, -127, 127, out=y_rounded)
        y_q = y_rounded.astype(np.int8)

        return y_q.astype(np.float32) * np.float32(self.output_scale)
