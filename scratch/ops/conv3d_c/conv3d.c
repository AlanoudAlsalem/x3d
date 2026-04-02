/*
 * conv3d.c — Multithreaded 3D convolution for RISC-V PolarFire SoC
 *
 * Targets the 4× U54 application cores (RV64GC, NO vector extension).
 * Performance levers: pthreads (4 threads), spatial tiling for L1 (32 KiB),
 * loop-unroll pragmas, and -O3 / -ffast-math compiler optimisation.
 *
 * Three internal fast paths selected at runtime:
 *   1. Pointwise  (kT==1 && kH==1 && kW==1)
 *   2. Depthwise  (groups == C_out && groups > 1)
 *   3. General    (everything else)
 *
 * All paths partition (batch × output-channel) work items across threads.
 * Each thread writes to non-overlapping output slices — no synchronisation
 * is needed beyond join.
 */

#include "conv3d.h"
#include <pthread.h>

#define NUM_THREADS 4

/* Spatial tile sizes chosen so that the input receptive field of one tile
   fits comfortably inside the U54's 32 KiB L1 data cache.              */
#define TILE_H 8
#define TILE_W 16

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* ================================================================== */
/*  Helpers                                                           */
/* ================================================================== */

/* Flat index into a contiguous 5-D tensor with shape [D0, D1, D2, D3, D4]. */
static inline long idx5(int i0, int i1, int i2, int i3, int i4,
                        int d1, int d2, int d3, int d4)
{
    return ((((long)i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4 + i4;
}

/*
 * Read from NCTHW input with implicit zero-padding.
 * (t, h, w) are in *padded* coordinates.  The function subtracts the
 * padding offsets to recover the actual position and returns 0.0f for
 * any out-of-bounds access.  The unsigned cast folds the < 0 and >= dim
 * checks into a single comparison.
 */
static inline float padval(const float *x,
                           int b, int c, int t, int h, int w,
                           int C, int T, int H, int W,
                           int pt, int ph, int pw)
{
    int ti = t - pt, hi = h - ph, wi = w - pw;
    if ((unsigned)ti >= (unsigned)T ||
        (unsigned)hi >= (unsigned)H ||
        (unsigned)wi >= (unsigned)W)
        return 0.f;
    return x[idx5(b, c, ti, hi, wi, C, T, H, W)];
}

/* ================================================================== */
/*  Per-thread argument block                                         */
/* ================================================================== */

typedef struct {
    const float *in;     /* input  [B, Ci, T, H, W]             */
    const float *wt;     /* weight [Co, cpg, kT, kH, kW]        */
    const float *bi;     /* bias   [Co]  (may be NULL)           */
    float       *out;    /* output [B, Co, To, Ho, Wo]           */

    int B, Ci, T, H, W;
    int Co, kT, kH, kW;
    int st, sh, sw;
    int pt, ph, pw;
    int groups, cpg;
    int To, Ho, Wo;

    int task_lo, task_hi; /* range of  b*Co + oc  to process     */
} targ_t;

/* Forward declarations for the three fast paths. */
static void pw_work (targ_t *a);
static void dw_work (targ_t *a);
static void gen_work(targ_t *a);

/* ================================================================== */
/*  Fast path 1 — Pointwise (1×1×1 kernel)                           */
/*                                                                    */
/*  No spatial kernel overlap → sequential spatial access is already  */
/*  cache-friendly.  The hot loop is the channel accumulation.        */
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
        const float bv = a->bi ? a->bi[oc] : 0.f;
        /* weight layout: [Co, cpg, 1, 1, 1] → cpg contiguous floats */
        const float *kp = a->wt + (long)oc * cpg;

        for (int to = 0; to < To; to++) {
            const int ti = to * st - pt;
            for (int ho = 0; ho < Ho; ho++) {
                const int hi = ho * sh - ph;
                const long obase = idx5(b, oc, to, ho, 0, Co, To, Ho, Wo);

                /* If temporal or height falls in the padding zone,
                   every output in this row is just the bias.        */
                if ((unsigned)ti >= (unsigned)T ||
                    (unsigned)hi >= (unsigned)H) {
                    for (int wo = 0; wo < Wo; wo++)
                        a->out[obase + wo] = bv;
                    continue;
                }

                #pragma GCC unroll 4
                for (int wo = 0; wo < Wo; wo++) {
                    const int wi = wo * sw - pw;
                    if ((unsigned)wi >= (unsigned)W) {
                        a->out[obase + wo] = bv;
                        continue;
                    }
                    float s = bv;
                    #pragma GCC unroll 8
                    for (int c = 0; c < cpg; c++)
                        s += kp[c] *
                             a->in[idx5(b, cs + c, ti, hi, wi,
                                        Ci, T, H, W)];
                    a->out[obase + wo] = s;
                }
            }
        }
    }
}

/* ================================================================== */
/*  Fast path 2 — Depthwise (groups == Co, cpg == 1)                  */
/*                                                                    */
/*  Each output channel reads exactly one input channel.  The kernel  */
/*  is small (typically 3×3×3 or 5×1×1).  Spatial tiling keeps the   */
/*  input receptive field in L1.                                      */
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
        const float bv = a->bi ? a->bi[oc] : 0.f;
        const float *kp = a->wt + (long)oc * kT * kHW;

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
                            float s = bv;

                            #pragma GCC unroll 5
                            for (int dt = 0; dt < kT; dt++) {
                                const int tp = to * st + dt;
                                #pragma GCC unroll 3
                                for (int dh = 0; dh < kH; dh++) {
                                    const int hp = ho * sh + dh;
                                    #pragma GCC unroll 3
                                    for (int dw = 0; dw < kW; dw++) {
                                        const int wpc = wo * sw + dw;
                                        s += kp[dt * kHW + dh * kW + dw] *
                                             padval(a->in, b, ic,
                                                    tp, hp, wpc,
                                                    Ci, T, H, W,
                                                    pt, ph, pw);
                                    }
                                }
                            }
                            a->out[obase + wo] = s;
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================== */
/*  General convolution (any groups / kernel size)                     */
/*                                                                    */
/*  Handles the stem's (1,3,3) standard conv, and any future kernel   */
/*  shapes.  Spatial tiling + channel-first accumulation.             */
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
        const float bv = a->bi ? a->bi[oc] : 0.f;
        const float *ocw = a->wt + (long)oc * cpg * kTHW;

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
                            float s = bv;
                            for (int c = 0; c < cpg; c++) {
                                const float *ckp = ocw + (long)c * kTHW;
                                #pragma GCC unroll 3
                                for (int dt = 0; dt < kT; dt++) {
                                    const int tp = to * st + dt;
                                    #pragma GCC unroll 3
                                    for (int dh = 0; dh < kH; dh++) {
                                        const int hp = ho * sh + dh;
                                        #pragma GCC unroll 3
                                        for (int dw = 0; dw < kW; dw++) {
                                            const int wpc = wo * sw + dw;
                                            s += ckp[dt * kHW + dh * kW + dw] *
                                                 padval(a->in, b, cs + c,
                                                        tp, hp, wpc,
                                                        Ci, T, H, W,
                                                        pt, ph, pw);
                                        }
                                    }
                                }
                            }
                            a->out[obase + wo] = s;
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

void conv3d_forward_c(
    const float *input,
    const float *weight,
    const float *bias,
    float       *output,
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
            .in = input,  .wt = weight, .bi = bias, .out = output,
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
