/*
 * main.c — Compare the software and FPGA implementations of the minimal
 *          int8 3D convolution. Both functions have the SAME signature;
 *          this test feeds them identical inputs and verifies the outputs
 *          match element-for-element.
 *
 * Build:  make
 * Run  :  ./test_conv3d_simple
 *
 * The test uses a small, reproducible tensor so you can eyeball the
 * numbers if something goes wrong. Tweak the shape constants at the top
 * of main() to exercise different layer configurations.
 */

#include "conv3d_simple.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tiny deterministic PRNG so we don't depend on rand() behaviour. */
static uint32_t lcg_state = 0xC0FFEEu;
static int8_t rand_int8(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    /* Map to [-127, 127]. */
    int v = (int)((lcg_state >> 16) & 0xFF) - 128;
    if (v < -127) v = -127;
    if (v >  127) v =  127;
    return (int8_t)v;
}

int main(void) {
    /* Small conv_b-like shape: depthwise 3x3x3, stride 1, pad 1. */
    const int B      = 1;
    const int C_in   = 4;
    const int T      = 4;
    const int H      = 8;
    const int W      = 8;
    const int C_out  = 4;
    const int kT = 3, kH = 3, kW = 3;
    const int stride_t = 1, stride_h = 1, stride_w = 1;
    const int pad_t    = 1, pad_h    = 1, pad_w    = 1;
    const int groups   = 4;             /* depthwise */

    const int T_out = (T + 2 * pad_t - kT) / stride_t + 1;
    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    const size_t n_in  = (size_t)B * C_in * T * H * W;
    const size_t n_w   = (size_t)C_out * (C_in / groups) * kT * kH * kW;
    const size_t n_out = (size_t)B * C_out * T_out * H_out * W_out;

    int8_t  *input    = (int8_t  *)malloc(n_in  * sizeof(int8_t));
    int8_t  *weight   = (int8_t  *)malloc(n_w   * sizeof(int8_t));
    int32_t *out_sw   = (int32_t *)malloc(n_out * sizeof(int32_t));
    int32_t *out_fpga = (int32_t *)malloc(n_out * sizeof(int32_t));
    if (!input || !weight || !out_sw || !out_fpga) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < n_in; ++i) input[i]  = rand_int8();
    for (size_t i = 0; i < n_w;  ++i) weight[i] = rand_int8();
    memset(out_sw,   0, n_out * sizeof(int32_t));
    memset(out_fpga, 0, n_out * sizeof(int32_t));

    printf("input shape  : (%d, %d, %d, %d, %d)\n",  B, C_in, T, H, W);
    printf("weight shape : (%d, %d, %d, %d, %d)\n",
           C_out, C_in / groups, kT, kH, kW);
    printf("output shape : (%d, %d, %d, %d, %d)\n",
           B, C_out, T_out, H_out, W_out);
    printf("groups       : %d\n\n", groups);

    int rc_sw = conv3d_int8_sw(input, weight, out_sw,
                               B, C_in, T, H, W,
                               C_out, kT, kH, kW,
                               stride_t, stride_h, stride_w,
                               pad_t, pad_h, pad_w, groups);
    printf("conv3d_int8_sw   returned %d\n", rc_sw);

    int rc_fp = conv3d_int8_fpga(input, weight, out_fpga,
                                 B, C_in, T, H, W,
                                 C_out, kT, kH, kW,
                                 stride_t, stride_h, stride_w,
                                 pad_t, pad_h, pad_w, groups);
    printf("conv3d_int8_fpga returned %d\n\n", rc_fp);

    if (rc_sw != 0 || rc_fp != 0) {
        fprintf(stderr, "one of the implementations failed\n");
        return 1;
    }

    /* Element-wise compare. */
    size_t n_mismatch = 0;
    int32_t max_abs_diff = 0;
    for (size_t i = 0; i < n_out; ++i) {
        int32_t d = out_sw[i] - out_fpga[i];
        int32_t ad = d < 0 ? -d : d;
        if (ad != 0) ++n_mismatch;
        if (ad > max_abs_diff) max_abs_diff = ad;
    }

    printf("total output elements : %zu\n", n_out);
    printf("mismatched elements   : %zu\n", n_mismatch);
    printf("max abs diff          : %d\n", max_abs_diff);

    /* Show the first few values side-by-side for a sanity check. */
    printf("\nfirst 8 output values (sw vs fpga):\n");
    const size_t show = n_out < 8 ? n_out : 8;
    for (size_t i = 0; i < show; ++i) {
        printf("  [%zu]  sw=%-8d  fpga=%-8d  %s\n",
               i, out_sw[i], out_fpga[i],
               out_sw[i] == out_fpga[i] ? "ok" : "MISMATCH");
    }

    int passed = (n_mismatch == 0);
    printf("\n%s\n", passed ? "PASS: outputs match bit-for-bit."
                             : "FAIL: outputs differ.");

    free(input); free(weight); free(out_sw); free(out_fpga);
    return passed ? 0 : 1;
}
