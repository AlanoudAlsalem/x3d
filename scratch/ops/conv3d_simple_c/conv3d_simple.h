/*
 * conv3d_simple.h — Minimal int8 3D convolution API.
 *
 * No quantization. No requantization. No threading. No bias.
 * Just:  output[int32] = conv3d(input[int8], weight[int8])
 *
 * Two implementations share this header:
 *
 *   conv3d_int8_sw   — pure-C CPU reference. Nested loops.
 *   conv3d_int8_fpga — FPGA-offload entry point. Today it is a stub that
 *                      pretends to DMA data to an accelerator and reads it
 *                      back; when the real fabric lands, only the body of
 *                      this function changes. The signature is identical
 *                      to the software version so callers can swap them
 *                      with a function pointer.
 *
 * Tensor layouts (all contiguous, NCTHW convention from the Python side):
 *
 *   input  : [B,  C_in,      T,  H,  W ]              int8
 *   weight : [C_out, C_in/groups, kT, kH, kW]         int8
 *   output : [B,  C_out,     T_out, H_out, W_out]     int32 (caller-allocated)
 *
 *   T_out = (T + 2*pad_t - kT) / stride_t + 1
 *   H_out = (H + 2*pad_h - kH) / stride_h + 1
 *   W_out = (W + 2*pad_w - kW) / stride_w + 1
 *
 *   groups = 1           -> standard convolution
 *   groups = C_in = C_out -> depthwise
 */

#ifndef CONV3D_SIMPLE_H
#define CONV3D_SIMPLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Return codes for both implementations:
 *    0  success
 *   -1  invalid argument (NULL pointer, bad groups, bad shape)
 *   -2  FPGA unavailable / offload failed (only from conv3d_int8_fpga)
 */

int conv3d_int8_sw(
    const int8_t *input,
    const int8_t *weight,
    int32_t      *output,
    int B,  int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t,    int pad_h,    int pad_w,
    int groups
);

int conv3d_int8_fpga(
    const int8_t *input,
    const int8_t *weight,
    int32_t      *output,
    int B,  int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t,    int pad_h,    int pad_w,
    int groups
);

#ifdef __cplusplus
}
#endif

#endif /* CONV3D_SIMPLE_H */
