/*
 * conv3d_fpga.c — Int8 3D convolution with fixed-point requantization
 *
 * Mirrors the FPGA accelerator's datapath: int8×int8 → int32 accumulate →
 * fixed-point (M0, n) requantize → int8 output.
 *
 * Same threading model as conv3d_c/conv3d.c: pthreads with NUM_THREADS
 * workers, each processing a contiguous chunk of (batch, output-channel)
 * pairs. Three fast paths: pointwise, depthwise, general.
 *
 * Targets RV64GC (PolarFire SoC, 4× U54 cores, 32 KiB L1, no vector ext).
 */

#include "conv3d_fpga.h"
#include <pthread.h>

#define NUM_THREADS 4

#define TILE_H 8
#define TILE_W 16

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline int8_t clip_int8(int64_t v)
{
    if (v > 127)  return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

static inline int8_t requantize(int32_t acc, int64_t m0, int32_t shift)
{
    int64_t prod = (int64_t)acc * m0;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t sign = (prod >= 0) ? 1 : -1;
    int64_t rounded = (prod + sign * half) >> shift;
    return clip_int8(rounded);
}

/* Flat index into contiguous 5-D tensor [D0, D1, D2, D3, D4]. */
static inline long idx5(int i0, int i1, int i2, int i3, int i4,
                        int d1, int d2, int d3, int d4)
{
    return ((((long)i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4 + i4;
}

/*
 * Read from NCTHW int8 input with implicit zero-padding.
 * Returns 0 for out-of-bounds accesses.
 */
static inline int8_t padval_i8(const int8_t *x,
                                int b, int c, int t, int h, int w,
                                int C, int T, int H, int W,
                                int pt, int ph, int pw)
{
    int ti = t - pt, hi = h - ph, wi = w - pw;
    if ((unsigned)ti >= (unsigned)T ||
        (unsigned)hi >= (unsigned)H ||
        (unsigned)wi >= (unsigned)W)
        return 0;
    return x[idx5(b, c, ti, hi, wi, C, T, H, W)];
}

/* ================================================================== */
/*  Per-thread argument block                                         */
/* ================================================================== */

typedef struct {
    const int8_t  *in;
    const int8_t  *wt;
    int8_t        *out;
    const int64_t *M0;
    const int32_t *n;

    int B, Ci, T, H, W;
    int Co, kT, kH, kW;
    int st, sh, sw;
    int pt, ph, pw;
    int groups, cpg;
    int To, Ho, Wo;

    int task_lo, task_hi;
} targ_t;

static void pw_work(targ_t *a);
static void dw_work(targ_t *a);
static void gen_work(targ_t *a);

/* ================================================================== */
/*  Fast path 1 — Pointwise (1×1×1)                                  */
/* ================================================================== */

static void pw_work(targ_t *a)
{
    const int Ci  = a->Ci, Co = a->Co;
    const int T   = a->T,  H  = a->H,  W  = a->W;
    const int To  = a->To, Ho = a->Ho, Wo = a->Wo;
    const int st  = a->st, sh = a->sh, sw = a->sw;
    const int pt  = a->pt, ph = a->ph, pw = a->pw;
    const int cpg = a->cpg, groups = a->groups;

    for (int task = a->task_lo; task < a->task_hi; task++) {
        const int b  = task / Co;
        const int oc = task % Co;
        const int g  = oc % groups;
        const int cs = g * cpg;
        const int64_t m0 = a->M0[oc];
        const int32_t ns = a->n[oc];
        const int8_t *kp = a->wt + (long)oc * cpg;

        for (int to = 0; to < To; to++) {
            const int ti = to * st - pt;
            for (int ho = 0; ho < Ho; ho++) {
                const int hi = ho * sh - ph;
                const long obase = idx5(b, oc, to, ho, 0, Co, To, Ho, Wo);

                if ((unsigned)ti >= (unsigned)T ||
                    (unsigned)hi >= (unsigned)H) {
                    for (int wo = 0; wo < Wo; wo++)
                        a->out[obase + wo] = requantize(0, m0, ns);
                    continue;
                }

                #pragma GCC unroll 4
                for (int wo = 0; wo < Wo; wo++) {
                    const int wi = wo * sw - pw;
                    if ((unsigned)wi >= (unsigned)W) {
                        a->out[obase + wo] = requantize(0, m0, ns);
                        continue;
                    }
                    int32_t acc = 0;
                    #pragma GCC unroll 8
                    for (int c = 0; c < cpg; c++)
                        acc += (int32_t)kp[c] *
                               (int32_t)a->in[idx5(b, cs + c, ti, hi, wi,
                                                    Ci, T, H, W)];
                    a->out[obase + wo] = requantize(acc, m0, ns);
                }
            }
        }
    }
}

/* ================================================================== */
/*  Fast path 2 — Depthwise (groups == Co, cpg == 1)                  */
/* ================================================================== */

static void dw_work(targ_t *a)
{
    const int Ci  = a->Ci, Co = a->Co;
    const int T   = a->T,  H  = a->H,  W  = a->W;
    const int To  = a->To, Ho = a->Ho, Wo = a->Wo;
    const int kT  = a->kT, kH = a->kH, kW = a->kW;
    const int st  = a->st, sh = a->sh, sw = a->sw;
    const int pt  = a->pt, ph = a->ph, pw = a->pw;
    const int kHW = kH * kW;

    for (int task = a->task_lo; task < a->task_hi; task++) {
        const int b  = task / Co;
        const int oc = task % Co;
        const int ic = oc;
        const int64_t m0 = a->M0[oc];
        const int32_t ns = a->n[oc];
        const int8_t *kp = a->wt + (long)oc * kT * kHW;

        for (int to = 0; to < To; to++) {
            for (int hb = 0; hb < Ho; hb += TILE_H) {
                const int he = MIN(hb + TILE_H, Ho);
                for (int wb = 0; wb < Wo; wb += TILE_W) {
                    const int we = MIN(wb + TILE_W, Wo);

                    for (int ho = hb; ho < he; ho++) {
                        const long obase = idx5(b, oc, to, ho, 0,
                                                Co, To, Ho, Wo);

                        #pragma GCC unroll 4
                        for (int wo = wb; wo < we; wo++) {
                            int32_t acc = 0;

                            #pragma GCC unroll 5
                            for (int dt = 0; dt < kT; dt++) {
                                const int tp = to * st + dt;
                                #pragma GCC unroll 3
                                for (int dh = 0; dh < kH; dh++) {
                                    const int hp = ho * sh + dh;
                                    #pragma GCC unroll 3
                                    for (int dw = 0; dw < kW; dw++) {
                                        const int wpc = wo * sw + dw;
                                        acc += (int32_t)kp[dt * kHW + dh * kW + dw] *
                                               (int32_t)padval_i8(a->in, b, ic,
                                                                   tp, hp, wpc,
                                                                   Ci, T, H, W,
                                                                   pt, ph, pw);
                                    }
                                }
                            }
                            a->out[obase + wo] = requantize(acc, m0, ns);
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================== */
/*  General convolution (any groups / kernel size)                     */
/* ================================================================== */

static void gen_work(targ_t *a)
{
    const int Ci  = a->Ci, Co = a->Co;
    const int T   = a->T,  H  = a->H,  W  = a->W;
    const int To  = a->To, Ho = a->Ho, Wo = a->Wo;
    const int kT  = a->kT, kH = a->kH, kW = a->kW;
    const int st  = a->st, sh = a->sh, sw = a->sw;
    const int pt  = a->pt, ph = a->ph, pw = a->pw;
    const int cpg = a->cpg, groups = a->groups;
    const int kHW  = kH * kW;
    const int kTHW = kT * kHW;

    for (int task = a->task_lo; task < a->task_hi; task++) {
        const int b  = task / Co;
        const int oc = task % Co;
        const int g  = oc % groups;
        const int cs = g * cpg;
        const int64_t m0 = a->M0[oc];
        const int32_t ns = a->n[oc];
        const int8_t *ocw = a->wt + (long)oc * cpg * kTHW;

        for (int to = 0; to < To; to++) {
            for (int hb = 0; hb < Ho; hb += TILE_H) {
                const int he = MIN(hb + TILE_H, Ho);
                for (int wb = 0; wb < Wo; wb += TILE_W) {
                    const int we = MIN(wb + TILE_W, Wo);

                    for (int ho = hb; ho < he; ho++) {
                        const long obase = idx5(b, oc, to, ho, 0,
                                                Co, To, Ho, Wo);

                        #pragma GCC unroll 4
                        for (int wo = wb; wo < we; wo++) {
                            int32_t acc = 0;
                            for (int c = 0; c < cpg; c++) {
                                const int8_t *ckp = ocw + (long)c * kTHW;
                                #pragma GCC unroll 3
                                for (int dt = 0; dt < kT; dt++) {
                                    const int tp = to * st + dt;
                                    #pragma GCC unroll 3
                                    for (int dh = 0; dh < kH; dh++) {
                                        const int hp = ho * sh + dh;
                                        #pragma GCC unroll 3
                                        for (int dw = 0; dw < kW; dw++) {
                                            const int wpc = wo * sw + dw;
                                            acc += (int32_t)ckp[dt * kHW + dh * kW + dw] *
                                                   (int32_t)padval_i8(a->in, b, cs + c,
                                                                      tp, hp, wpc,
                                                                      Ci, T, H, W,
                                                                      pt, ph, pw);
                                        }
                                    }
                                }
                            }
                            a->out[obase + wo] = requantize(acc, m0, ns);
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================== */
/*  Thread dispatch + public entry point                              */
/* ================================================================== */

static void *thread_fn(void *arg)
{
    targ_t *a = (targ_t *)arg;

    if (a->kT == 1 && a->kH == 1 && a->kW == 1)
        pw_work(a);
    else if (a->groups == a->Co && a->groups > 1)
        dw_work(a);
    else
        gen_work(a);

    return NULL;
}

void conv3d_fpga_int8(
    const int8_t  *input,
    const int8_t  *weight,
    int8_t        *output,
    const int64_t *M0,
    const int32_t *n,
    int B,  int Ci, int T,  int H,  int W,
    int Co, int kT, int kH, int kW,
    int st, int sh, int sw,
    int pt, int ph, int pw,
    int groups)
{
    const int Tp  = T + 2 * pt;
    const int Hp  = H + 2 * ph;
    const int Wp  = W + 2 * pw;
    const int To  = (Tp - kT) / st + 1;
    const int Ho  = (Hp - kH) / sh + 1;
    const int Wo  = (Wp - kW) / sw + 1;
    const int cpg = Ci / groups;

    const int total = B * Co;
    const int nt = total < NUM_THREADS ? total : NUM_THREADS;

    pthread_t thr[NUM_THREADS];
    targ_t    args[NUM_THREADS];
    const int per = (total + nt - 1) / nt;

    for (int i = 0; i < nt; i++) {
        int lo = i * per;
        int hi = (i + 1) * per;
        if (hi > total) hi = total;

        args[i] = (targ_t){
            .in = input,  .wt = weight, .out = output,
            .M0 = M0, .n = n,
            .B  = B,  .Ci = Ci, .T  = T,  .H  = H,  .W  = W,
            .Co = Co, .kT = kT, .kH = kH, .kW = kW,
            .st = st, .sh = sh, .sw = sw,
            .pt = pt, .ph = ph, .pw = pw,
            .groups = groups, .cpg = cpg,
            .To = To, .Ho = Ho, .Wo = Wo,
            .task_lo = lo, .task_hi = hi,
        };
        pthread_create(&thr[i], NULL, thread_fn, &args[i]);
    }

    for (int i = 0; i < nt; i++)
        pthread_join(thr[i], NULL);
}
