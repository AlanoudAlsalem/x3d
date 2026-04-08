/*
 * conv3d_fpga.h — Int8 3D convolution with fixed-point requantization
 *
 * Mirrors the FPGA accelerator's datapath in software:
 *   int8 input × int8 weights → int32 accumulator → (M0, n) requantize → int8 output
 *
 * Same threading model as the float32 C backend (conv3d_c/), same API conventions.
 * All tensors are contiguous NCTHW.
 */

#ifndef CONV3D_FPGA_H
#define CONV3D_FPGA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * conv3d_fpga_int8 — Int8 3D convolution + fixed-point requantization.
 *
 * Parameters
 * ----------
 * input   : [B, C_in, T, H, W]                (contiguous int8)
 * weight  : [C_out, C_in/groups, kT, kH, kW]  (contiguous int8)
 * output  : [B, C_out, T_out, H_out, W_out]   (pre-allocated int8)
 * M0      : [C_out]                            (per-channel int64 multiplier)
 * n       : [C_out]                            (per-channel int32 shift)
 *
 *   T_out = (T + 2*pad_t - kT) / stride_t + 1
 *   H_out = (H + 2*pad_h - kH) / stride_h + 1
 *   W_out = (W + 2*pad_w - kW) / stride_w + 1
 *
 * groups  : 1 for standard convolution, C_in for depthwise.
 *
 * Requantization per output element:
 *   acc32 = sum_over_kernel(input[int8] * weight[int8])   // int32 accumulator
 *   prod  = (int64)acc32 * M0[c]                          // int64 product
 *   half  = (n > 0) ? (1 << (n-1)) : 0                   // rounding bias
 *   y     = clip_int8( (prod + sign(prod)*half) >> n )    // round-half-away-from-zero
 */
void conv3d_fpga_int8(
    const int8_t  *input,
    const int8_t  *weight,
    int8_t        *output,
    const int64_t *M0,
    const int32_t *n,
    int B,  int C_in, int T,  int H,  int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w,
    int groups
);

#ifdef __cplusplus
}
#endif

#endif /* CONV3D_FPGA_H */
