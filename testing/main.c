/*
 * main.c — Compare the software and FPGA implementations of the minimal
 *          int8 3D convolution across ALL 28 unique Conv3d configurations
 *          in X3D-M. Both functions have the SAME signature; this test
 *          feeds them identical inputs and verifies the outputs match
 *          element-for-element.
 *
 * Build:  make
 * Run  :  ./test_conv3d_simple
 */

#include "conv3d_simple.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Layer configuration: one entry per unique Conv3d in X3D-M.          */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    int B, C_in, T, H, W;
    int C_out;
    int kT, kH, kW;
    int stride_t, stride_h, stride_w;
    int pad_t, pad_h, pad_w;
    int groups;
} LayerCfg;

static const LayerCfg LAYERS[] = {
    /* ---- Stem ---- */
    /*  0 */ {"conv_t",                    1,   3, 16, 224, 224,   24,  1,3,3,  1,2,2,  0,1,1,    1},
    /*  1 */ {"conv_xy",                   1,  24, 16, 112, 112,   24,  5,1,1,  1,1,1,  2,0,0,   24},

    /* ---- Branch1 skip connections (first block of each stage) ---- */
    /*  2 */ {"branch1_s2",                1,  24, 16, 112, 112,   24,  1,1,1,  1,2,2,  0,0,0,    1},
    /*  3 */ {"branch1_s3",                1,  24, 16,  56,  56,   48,  1,1,1,  1,2,2,  0,0,0,    1},
    /*  4 */ {"branch1_s4",                1,  48, 16,  28,  28,   96,  1,1,1,  1,2,2,  0,0,0,    1},
    /*  5 */ {"branch1_s5",                1,  96, 16,  14,  14,  192,  1,1,1,  1,2,2,  0,0,0,    1},

    /* ---- conv_a (1x1x1 expand) ---- */
    /*  6 */ {"conv_a_s2_blk0",            1,  24, 16, 112, 112,   54,  1,1,1,  1,1,1,  0,0,0,    1},
    /*  7 */ {"conv_a_s2_blk1-2",          1,  24, 16,  56,  56,   54,  1,1,1,  1,1,1,  0,0,0,    1},
    /*  8 */ {"conv_a_s3_blk0",            1,  24, 16,  56,  56,  108,  1,1,1,  1,1,1,  0,0,0,    1},
    /*  9 */ {"conv_a_s3_blk1-4",          1,  48, 16,  28,  28,  108,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 10 */ {"conv_a_s4_blk0",            1,  48, 16,  28,  28,  216,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 11 */ {"conv_a_s4_blk1-10",         1,  96, 16,  14,  14,  216,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 12 */ {"conv_a_s5_blk0",            1,  96, 16,  14,  14,  432,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 13 */ {"conv_a_s5_blk1-6",          1, 192, 16,   7,   7,  432,  1,1,1,  1,1,1,  0,0,0,    1},

    /* ---- conv_b (3x3x3 depthwise) ---- */
    /* 14 */ {"conv_b_s2_blk0",            1,  54, 16, 112, 112,   54,  3,3,3,  1,2,2,  1,1,1,   54},
    /* 15 */ {"conv_b_s2_blk1-2",          1,  54, 16,  56,  56,   54,  3,3,3,  1,1,1,  1,1,1,   54},
    /* 16 */ {"conv_b_s3_blk0",            1, 108, 16,  56,  56,  108,  3,3,3,  1,2,2,  1,1,1,  108},
    /* 17 */ {"conv_b_s3_blk1-4",          1, 108, 16,  28,  28,  108,  3,3,3,  1,1,1,  1,1,1,  108},
    /* 18 */ {"conv_b_s4_blk0",            1, 216, 16,  28,  28,  216,  3,3,3,  1,2,2,  1,1,1,  216},
    /* 19 */ {"conv_b_s4_blk1-10",         1, 216, 16,  14,  14,  216,  3,3,3,  1,1,1,  1,1,1,  216},
    /* 20 */ {"conv_b_s5_blk0",            1, 432, 16,  14,  14,  432,  3,3,3,  1,2,2,  1,1,1,  432},
    /* 21 */ {"conv_b_s5_blk1-6",          1, 432, 16,   7,   7,  432,  3,3,3,  1,1,1,  1,1,1,  432},

    /* ---- conv_c (1x1x1 project) ---- */
    /* 22 */ {"conv_c_s2",                 1,  54, 16,  56,  56,   24,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 23 */ {"conv_c_s3",                 1, 108, 16,  28,  28,   48,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 24 */ {"conv_c_s4",                 1, 216, 16,  14,  14,   96,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 25 */ {"conv_c_s5",                 1, 432, 16,   7,   7,  192,  1,1,1,  1,1,1,  0,0,0,    1},

    /* ---- Head ---- */
    /* 26 */ {"head_pre_conv",             1, 192, 16,   7,   7,  432,  1,1,1,  1,1,1,  0,0,0,    1},
    /* 27 */ {"head_post_conv",            1, 432,  1,   1,   1, 2048,  1,1,1,  1,1,1,  0,0,0,    1},
};

#define NUM_LAYERS (sizeof(LAYERS) / sizeof(LAYERS[0]))

/* ------------------------------------------------------------------ */
/* Tiny deterministic PRNG so we don't depend on rand() behaviour.     */
/* ------------------------------------------------------------------ */

static uint32_t lcg_state;
static int8_t rand_int8(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    int v = (int)((lcg_state >> 16) & 0xFF) - 128;
    if (v < -127) v = -127;
    if (v >  127) v =  127;
    return (int8_t)v;
}

/* ------------------------------------------------------------------ */

int main(void) {
    int total_pass = 0;
    int total_fail = 0;

    for (int idx = 0; idx < (int)NUM_LAYERS; idx++) {
        const LayerCfg *L = &LAYERS[idx];

        const int T_out = (L->T + 2 * L->pad_t - L->kT) / L->stride_t + 1;
        const int H_out = (L->H + 2 * L->pad_h - L->kH) / L->stride_h + 1;
        const int W_out = (L->W + 2 * L->pad_w - L->kW) / L->stride_w + 1;

        const size_t n_in  = (size_t)L->B * L->C_in * L->T * L->H * L->W;
        const size_t n_w   = (size_t)L->C_out * (L->C_in / L->groups)
                             * L->kT * L->kH * L->kW;
        const size_t n_out = (size_t)L->B * L->C_out * T_out * H_out * W_out;

        int8_t  *input    = (int8_t  *)malloc(n_in  * sizeof(int8_t));
        int8_t  *weight   = (int8_t  *)malloc(n_w   * sizeof(int8_t));
        int32_t *out_sw   = (int32_t *)malloc(n_out * sizeof(int32_t));
        int32_t *out_fpga = (int32_t *)malloc(n_out * sizeof(int32_t));
        if (!input || !weight || !out_sw || !out_fpga) {
            fprintf(stderr, "[%2d] %s: allocation failed\n", idx, L->name);
            free(input); free(weight); free(out_sw); free(out_fpga);
            total_fail++;
            continue;
        }

        /* Reset PRNG per layer so each test is independent. */
        lcg_state = 0xC0FFEEu + (uint32_t)idx;
        for (size_t j = 0; j < n_in; ++j) input[j]  = rand_int8();
        for (size_t j = 0; j < n_w;  ++j) weight[j] = rand_int8();
        memset(out_sw,   0, n_out * sizeof(int32_t));
        memset(out_fpga, 0, n_out * sizeof(int32_t));

        printf("──────────────────────────────────────────────────\n");
        printf("[%2d] %s\n", idx, L->name);
        printf("     input  (%d, %d, %d, %d, %d)\n",
               L->B, L->C_in, L->T, L->H, L->W);
        printf("     weight (%d, %d, %d, %d, %d)  groups=%d\n",
               L->C_out, L->C_in / L->groups, L->kT, L->kH, L->kW,
               L->groups);
        printf("     output (%d, %d, %d, %d, %d)\n",
               L->B, L->C_out, T_out, H_out, W_out);

        int rc_sw = conv3d_int8_sw(input, weight, out_sw,
                                   L->B, L->C_in, L->T, L->H, L->W,
                                   L->C_out, L->kT, L->kH, L->kW,
                                   L->stride_t, L->stride_h, L->stride_w,
                                   L->pad_t, L->pad_h, L->pad_w,
                                   L->groups);

        int rc_fp = conv3d_int8_fpga(input, weight, out_fpga,
                                     L->B, L->C_in, L->T, L->H, L->W,
                                     L->C_out, L->kT, L->kH, L->kW,
                                     L->stride_t, L->stride_h, L->stride_w,
                                     L->pad_t, L->pad_h, L->pad_w,
                                     L->groups);

        if (rc_sw != 0 || rc_fp != 0) {
            printf("     ERROR: sw returned %d, fpga returned %d\n",
                   rc_sw, rc_fp);
            total_fail++;
            free(input); free(weight); free(out_sw); free(out_fpga);
            continue;
        }

        /* Element-wise compare. */
        size_t n_mismatch = 0;
        int32_t max_abs_diff = 0;
        for (size_t j = 0; j < n_out; ++j) {
            int32_t d = out_sw[j] - out_fpga[j];
            int32_t ad = d < 0 ? -d : d;
            if (ad != 0) ++n_mismatch;
            if (ad > max_abs_diff) max_abs_diff = ad;
        }

        int passed = (n_mismatch == 0);
        printf("     elements=%zu  mismatched=%zu  max_diff=%d  %s\n",
               n_out, n_mismatch, max_abs_diff,
               passed ? "PASS" : "FAIL");

        if (passed) total_pass++;
        else        total_fail++;

        free(input); free(weight); free(out_sw); free(out_fpga);
    }

    printf("══════════════════════════════════════════════════\n");
    printf("TOTAL: %d passed, %d failed out of %d layers\n",
           total_pass, total_fail, (int)NUM_LAYERS);
    printf("%s\n", total_fail == 0 ? "ALL PASS" : "SOME FAILED");
    return total_fail == 0 ? 0 : 1;
}
