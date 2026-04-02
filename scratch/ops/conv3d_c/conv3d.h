/*
 * conv3d.h — Multithreaded 3D convolution for RISC-V PolarFire SoC
 *
 * Single entry point that internally dispatches to fast paths for
 * pointwise (1×1×1), depthwise, and general convolutions.
 *
 * All tensors are float32, layout NCTHW.
 */

#ifndef CONV3D_H
#define CONV3D_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * conv3d_forward_c — 3D convolution with implicit zero-padding.
 *
 * Parameters
 * ----------
 * input   : [B, C_in, T, H, W]                      (contiguous float32)
 * weight  : [C_out, C_in/groups, kT, kH, kW]         (contiguous float32)
 * bias    : [C_out]  or NULL                          (contiguous float32)
 * output  : [B, C_out, T_out, H_out, W_out]          (pre-allocated float32)
 *
 *   T_out = (T + 2*pad_t - kT) / stride_t + 1
 *   H_out = (H + 2*pad_h - kH) / stride_h + 1
 *   W_out = (W + 2*pad_w - kW) / stride_w + 1
 *
 * groups  : 1 for standard convolution, C_in for depthwise.
 */
void conv3d_forward_c(
    const float *input,
    const float *weight,
    const float *bias,
    float       *output,
    int B,  int C_in, int T,  int H,  int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w,
    int groups
);

#ifdef __cplusplus
}
#endif

#endif /* CONV3D_H */
