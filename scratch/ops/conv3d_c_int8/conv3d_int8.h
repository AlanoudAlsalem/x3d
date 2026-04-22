/*
 * conv3d_int8.h — Int8 multithreaded 3D convolution for RISC-V PolarFire SoC
 *
 * Mirrors conv3d_c/conv3d.h but operates on quantized int8 tensors:
 *   int8 input x int8 weights -> int32 accumulator -> int32 output
 *
 * The raw int32 accumulator is written directly to the output buffer —
 * no requantization is performed.
 *
 * Same threading model as the float32 C backend (conv3d_c/), same API conventions.
 * All tensors are contiguous NCTHW.
 */

#ifndef CONV3D_INT8_H
#define CONV3D_INT8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * conv3d_int8_forward_c — Int8 3D convolution with int32 output.
 *
 * Parameters
 * ----------
 * input   : [B, C_in, T, H, W]                (contiguous int8)
 * weight  : [C_out, C_in/groups, kT, kH, kW]  (contiguous int8)
 * output  : [B, C_out, T_out, H_out, W_out]   (pre-allocated int32)
 *
 *   T_out = (T + 2*pad_t - kT) / stride_t + 1
 *   H_out = (H + 2*pad_h - kH) / stride_h + 1
 *   W_out = (W + 2*pad_w - kW) / stride_w + 1
 *
 * groups  : 1 for standard convolution, C_in for depthwise.
 *
 * Each output element is the raw int32 dot-product accumulator:
 *   output[b,c,t,h,w] = sum_over_kernel(input[int8] * weight[int8])
 */
void conv3d_int8_forward_c(
    const int8_t  *input,
    const int8_t  *weight,
    int32_t       *output,
    int B,  int C_in, int T,  int H,  int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w,
    int groups
);

#ifdef __cplusplus
}
#endif

#endif /* CONV3D_INT8_H */
