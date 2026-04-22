/*
 * conv3d_sw.c — Software (CPU) reference for minimal int8 3D convolution.
 *
 * Single-threaded, deeply nested loops, no bias, no requantization.
 * The accumulator is int32. Promotion rules: int8 * int8 is first promoted
 * to int (at least 16-bit, usually 32-bit on our targets), then added into
 * the int32 accumulator — no overflow possible for any X3D-M layer.
 *
 * This file is the "ground truth" the FPGA implementation must match.
 */

#include "conv3d_simple.h"
#include <stddef.h>

int conv3d_int8_sw(
    const int8_t *input,
    const int8_t *weight,
    int32_t      *output,
    int B,  int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t,    int pad_h,    int pad_w,
    int groups)
{
    if (!input || !weight || !output) return -1;
    if (groups <= 0) return -1;
    if (C_in  % groups != 0) return -1;
    if (C_out % groups != 0) return -1;

    const int C_in_per_g  = C_in  / groups;
    const int C_out_per_g = C_out / groups;

    const int T_out = (T + 2 * pad_t - kT) / stride_t + 1;
    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    if (T_out <= 0 || H_out <= 0 || W_out <= 0) return -1;

    /* Strides in elements for NCTHW layouts. */
    const size_t in_stride_w = 1;
    const size_t in_stride_h = (size_t)W;
    const size_t in_stride_t = (size_t)H * W;
    const size_t in_stride_c = (size_t)T * H * W;
    const size_t in_stride_b = (size_t)C_in * T * H * W;

    const size_t out_stride_w = 1;
    const size_t out_stride_h = (size_t)W_out;
    const size_t out_stride_t = (size_t)H_out * W_out;
    const size_t out_stride_c = (size_t)T_out * H_out * W_out;
    const size_t out_stride_b = (size_t)C_out * T_out * H_out * W_out;

    const size_t w_stride_kw = 1;
    const size_t w_stride_kh = (size_t)kW;
    const size_t w_stride_kt = (size_t)kH * kW;
    const size_t w_stride_ci = (size_t)kT * kH * kW;
    const size_t w_stride_co = (size_t)C_in_per_g * kT * kH * kW;

    for (int b = 0; b < B; ++b) {
        for (int g = 0; g < groups; ++g) {
            const int co_start = g * C_out_per_g;
            const int ci_start = g * C_in_per_g;

            for (int co = 0; co < C_out_per_g; ++co) {
                const int c_out = co_start + co;

                for (int ot = 0; ot < T_out; ++ot) {
                    for (int oh = 0; oh < H_out; ++oh) {
                        for (int ow = 0; ow < W_out; ++ow) {

                            int32_t acc = 0;

                            for (int ci = 0; ci < C_in_per_g; ++ci) {
                                const int c_in = ci_start + ci;

                                for (int kt = 0; kt < kT; ++kt) {
                                    const int it = ot * stride_t + kt - pad_t;
                                    if (it < 0 || it >= T) continue;

                                    for (int kh = 0; kh < kH; ++kh) {
                                        const int ih = oh * stride_h + kh - pad_h;
                                        if (ih < 0 || ih >= H) continue;

                                        for (int kw = 0; kw < kW; ++kw) {
                                            const int iw = ow * stride_w + kw - pad_w;
                                            if (iw < 0 || iw >= W) continue;

                                            const size_t in_off =
                                                  b     * in_stride_b
                                                + c_in  * in_stride_c
                                                + it    * in_stride_t
                                                + ih    * in_stride_h
                                                + iw    * in_stride_w;

                                            const size_t w_off =
                                                  c_out * w_stride_co
                                                + ci    * w_stride_ci
                                                + kt    * w_stride_kt
                                                + kh    * w_stride_kh
                                                + kw    * w_stride_kw;

                                            acc += (int32_t)input[in_off]
                                                 * (int32_t)weight[w_off];
                                        }
                                    }
                                }
                            }

                            const size_t out_off =
                                  b     * out_stride_b
                                + c_out * out_stride_c
                                + ot    * out_stride_t
                                + oh    * out_stride_h
                                + ow    * out_stride_w;
                            output[out_off] = acc;
                        }
                    }
                }
            }
        }
    }
    return 0;
}
