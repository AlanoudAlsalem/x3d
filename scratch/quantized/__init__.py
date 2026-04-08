"""
scratch.quantized — int8 quantized runtime for X3D-M (FPGA-oriented).

This subpackage is completely isolated from the float32 path. It provides:

- A software reference int8 3D convolution kernel that mirrors the math the
  FPGA accelerator will perform (conv3d_int8).
- A QuantizedConv3d module that quantizes its float32 input, runs the int8
  conv, and dequantizes the int8 output back to float32.
- A loader for the int8 .npz files produced by scripts/quantize_x3d_ptq.py.
- A QuantizedX3D_M builder that constructs a float32 X3D_M skeleton and swaps
  every Conv3d for a QuantizedConv3d while folding BatchNorm3d away.

Nothing in this subpackage is imported by the float32 path. You can run
``scratch.models.x3d_m`` and ``scratch.quantized.model`` in the same Python
process on different model instances, and they will not interact.
"""

from scratch.quantized.conv3d_int8 import conv3d_int8_forward
from scratch.quantized.layers import QuantizedConv3d, QuantizedLinear
from scratch.quantized.load_int8_weights import load_int8_weights
from scratch.quantized.model import build_quantized_x3d_m, QuantizedX3D_M

__all__ = [
    "conv3d_int8_forward",
    "QuantizedConv3d",
    "QuantizedLinear",
    "load_int8_weights",
    "build_quantized_x3d_m",
    "QuantizedX3D_M",
]
