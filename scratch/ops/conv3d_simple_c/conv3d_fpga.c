/*
 * conv3d_fpga.c — FPGA-offload entry point for minimal int8 3D convolution.
 *
 * This file has the SAME function signature as conv3d_sw.c. That is the
 * whole point of the exercise: the caller does not care whether the
 * convolution ran on the CPU or on the FPGA fabric.
 *
 * TODAY (no real hardware): the body below is a stub that does three things
 * in sequence, mirroring what the real offload will do. All three steps are
 * clearly labelled so when the real PolarFire accelerator lands you only
 * have to replace the marked sections:
 *
 *     STEP 1  — DMA input + weights to FPGA-visible memory
 *     STEP 2  — Kick off the FPGA accelerator, wait for done
 *     STEP 3  — DMA the int32 output back to the host buffer
 *
 * For the stub, "the FPGA" is just a call to the software reference. That
 * is intentional: it guarantees the two implementations agree bit-for-bit
 * right now, so the compare harness in main.c exercises the full plumbing
 * without any "hardware" in the loop. When a real FPGA driver exists you
 * will get a real pass/fail signal from the same harness.
 */

#include "conv3d_simple.h"
#include <stddef.h>

/* Forward-declare the software reference so the stub can call it. */
int conv3d_int8_sw(
    const int8_t *input, const int8_t *weight, int32_t *output,
    int B, int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w,
    int groups);


/* ------------------------------------------------------------------ */
/* Placeholder DMA / control-register helpers.                         */
/* Replace these three functions when the real driver lands.           */
/* ------------------------------------------------------------------ */

static int fpga_dma_to_device(const void *host_ptr, size_t nbytes) {
    (void)host_ptr;
    (void)nbytes;
    /* TODO: real driver — memcpy to the DMA-mapped buffer, flush cache. */
    return 0;
}

static int fpga_launch_and_wait(
    int B, int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w,
    int groups)
{
    (void)B; (void)C_in; (void)T; (void)H; (void)W;
    (void)C_out; (void)kT; (void)kH; (void)kW;
    (void)stride_t; (void)stride_h; (void)stride_w;
    (void)pad_t; (void)pad_h; (void)pad_w; (void)groups;
    /* TODO: real driver — write shape regs, set "start" bit, poll "done". */
    return 0;
}

static int fpga_dma_from_device(void *host_ptr, size_t nbytes) {
    (void)host_ptr;
    (void)nbytes;
    /* TODO: real driver — invalidate cache, memcpy from DMA buffer. */
    return 0;
}

/* ------------------------------------------------------------------ */

int conv3d_int8_fpga(
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

    const size_t in_bytes =
        (size_t)B * C_in * T * H * W * sizeof(int8_t);
    const size_t w_bytes =
        (size_t)C_out * (C_in / (groups > 0 ? groups : 1))
                      * kT * kH * kW * sizeof(int8_t);

    /* STEP 1 — DMA input + weights into FPGA-visible memory. */
    if (fpga_dma_to_device(input,  in_bytes) != 0) return -2;
    if (fpga_dma_to_device(weight, w_bytes)  != 0) return -2;

    /* STEP 2 — Kick off the FPGA accelerator and wait for completion.
     *
     * TODAY this is a stub: we run the software reference directly so the
     * output buffer is filled with the mathematically correct result. When
     * the real fabric is wired up, delete the conv3d_int8_sw() call below
     * and let fpga_launch_and_wait() be the only thing that computes the
     * output.
     */
    if (fpga_launch_and_wait(B, C_in, T, H, W,
                             C_out, kT, kH, kW,
                             stride_t, stride_h, stride_w,
                             pad_t, pad_h, pad_w, groups) != 0) return -2;

    int rc = conv3d_int8_sw(input, weight, output,
                            B, C_in, T, H, W,
                            C_out, kT, kH, kW,
                            stride_t, stride_h, stride_w,
                            pad_t, pad_h, pad_w, groups);
    if (rc != 0) return rc;

    /* STEP 3 — DMA the int32 output back to the host buffer. */
    const int T_out = (T + 2 * pad_t - kT) / stride_t + 1;
    const int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    const int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    const size_t out_bytes =
        (size_t)B * C_out * T_out * H_out * W_out * sizeof(int32_t);
    if (fpga_dma_from_device(output, out_bytes) != 0) return -2;

    return 0;
}
