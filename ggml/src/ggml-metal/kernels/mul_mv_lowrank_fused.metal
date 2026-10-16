// filepath: latent_dflash/stage3_metal_engine/kernels/mul_mv_lowrank_fused.metal
//
// kernel_mul_mv_lowrank_fused_q4_k
//
//   y = A . (B . x)      B: [256 x D_in] Q4_K   A: [D_out x 256] Q4_K   x: FP32[D_in]   y: FP32[D_out]
//
// GGUF shapes (ne0 is the contiguous dim):  B = [D_in, 256]  ->  256 rows of D_in/256 blocks
//                                           A = [256, D_out] ->  D_out rows of exactly 1 block
//
// Geometry : 128 threads = 4 simdgroups. Grid = (n_tg, n_cols, 1). One threadgroup per (slice, column).
// Phase 1  : every threadgroup computes the full latent Z[0..255] = B.x (each simdgroup owns 64 rows,
//            LR_NR0 rows per pass, 32 lanes split every row over 4 blocks x 8 slices) and parks it in
//            1024 B of threadgroup memory. Z never touches device memory.
// Phase 2  : the threadgroup then grid-strides over groups of 16 rows of A. Each lane keeps its 32-value
//            slice of Z in registers (read once from threadgroup memory), streams one 16-byte qs slice of
//            each A row (8 lanes per row -> coalesced 128 B rows) and reduces the 8 lanes with xor-shuffles
//            after all loads of the iteration have been issued.
//
// Metal has no grid-wide barrier, so every threadgroup recomputes Z redundantly. That cost grows with n_tg;
// pick n_tg ~ 1-2 per GPU core (bench_lowrank_metal sweeps it).
//
// Math: all multiply-accumulates are FP32 fma. Integer ops are used only to extract the 4-bit fields and
// the 6-bit scale/min fields (no int8 dot, no packed-int SWAR accumulation, no half arithmetic).
//   w = d*sc*q - dmin*mn  =>  sum_k w_k z_k = d*sc*sum(q_k z_k) - dmin*mn*sum(z_k)   (per 32-value sub-block)
// High nibbles are consumed as (q<<4) and the 1/16 is folded into the scale.
//
// Registers: written to stay at or under 64 per thread; verify with pipeline.maxTotalThreadsPerThreadgroup
// (the bench prints it). If it reports < 128 or occupancy drops, rebuild with -DLR_NR0=2 and/or -DLR_RB=1.

#if defined(LOWRANK_FUSED_STANDALONE)

// Self-contained build used by bench_lowrank_metal.mm (runtime compile has no include path).
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K/2];
} block_q4_K;

// verbatim from dequantize.h
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// must match ggml-metal-lowrank.h
#define LOWRANK_FUSED_RANK      256
#define LOWRANK_FUSED_NTHREADS  128
#define LOWRANK_FUSED_SMEM      1024
typedef struct {
    int32_t  ne00;
    int32_t  ne0;
    int32_t  ne11;
    int32_t  _pad;
    uint64_t nb01;
    uint64_t nb1A;
    uint64_t nb11;
    uint64_t nb1;
} ggml_metal_kargs_mul_mv_lowrank_fused;

#else

#include "common.h"
#include "dequantize.h"
#include "ggml-metal-lowrank.h"

#endif

#ifndef LR_NR0
#define LR_NR0 4        // phase-1 rows per simdgroup pass (must divide 64)
#endif
#ifndef LR_RB
#define LR_RB  2        // phase-2 row groups in flight per lane
#endif
#define LR_NSG          (LOWRANK_FUSED_NTHREADS / 32)
#define LR_ROWS_PER_SG  (LOWRANK_FUSED_RANK / LR_NSG)

static inline float lr_sum4(float4 v) {
    return (v[0] + v[1]) + (v[2] + v[3]);
}

// threadgroup layout of Z: float4 slot q = k/4 is stored at q ^ (q>>4). With this the 8 distinct lane slices of
// phase 2 (stride 16/4 float4) land on 8 distinct 16-byte bank slots, and the 4 simdgroup leaders writing
// rows 64*sg + c in phase 1 land on 4 distinct banks (checked offline against a 32 x 4 B bank model).
static inline int lr_swz(int q) {
    return q ^ (q >> 4);
}

static inline int lr_zidx(int k) {
    return (lr_swz(k >> 2) << 2) | (k & 3);
}

[[max_total_threads_per_threadgroup(LOWRANK_FUSED_NTHREADS)]]
kernel void kernel_mul_mv_lowrank_fused_q4_k(
        constant ggml_metal_kargs_mul_mv_lowrank_fused & args,
        device const char * srcB,                           // Factor B  [D_in, 256]   Q4_K
        device const char * srcA,                           // Factor A  [256, D_out]  Q4_K
        device const char * src1,                           // x         [D_in, n_cols] F32
        device       char * dst,                            // y         [D_out, n_cols] F32
        threadgroup  char * shmem [[threadgroup(0)]],       // LOWRANK_FUSED_SMEM bytes
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tgpg  [[threadgroups_per_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup float4 * lds_Z4 = (threadgroup float4 *) shmem;     // 64 x float4 = 1024 B
    threadgroup float  * lds_Z  = (threadgroup float  *) shmem;

    const int nb = args.ne00 / QK_K;                                // Q4_K blocks per Factor-B row

    device const float * x     = (device const float *)(src1 + (ulong) tgpig.y * args.nb11);
    device       float * dst_f = (device       float *)(dst  + (ulong) tgpig.y * args.nb1);

    // ------------------------------------------------------------------------------------------------
    // Phase 1: Z[0..255] = B . x
    // lane -> (ix, iq, ir): ix = block slot (4 blocks in flight), iq = 128-value half, ir = 8-value chunk
    // each lane handles 32 values of a block: low/high nibbles of qs[32*iq + 8*ir + 0..7] and of
    // qs[64 + 32*iq + 8*ir + 0..7]  ->  sub-blocks 2*iq, 2*iq+1, 2*iq+4, 2*iq+5
    // ------------------------------------------------------------------------------------------------
    {
        const short ix = tiisg / 8;
        const short it = tiisg % 8;
        const short iq = it / 4;
        const short ir = it % 4;

        const uchar4 m0F = uchar4(0x0F);
        const uchar4 mF0 = uchar4(0xF0);

        for (short pass = 0; pass < LR_ROWS_PER_SG / LR_NR0; ++pass) {
            const int r0 = sgitg * LR_ROWS_PER_SG + pass * LR_NR0;

            device const char * brow = srcB + (ulong) r0 * args.nb01;

            float sumf[LR_NR0];
            FOR_UNROLL (short row = 0; row < LR_NR0; ++row) {
                sumf[row] = 0.f;
            }

            for (int ib = ix; ib < nb; ib += 4) {
                device const float4 * y4 = (device const float4 *)(x + ib * QK_K + 64 * iq + 8 * ir);

                const float4 yl0 = y4[0],  yl1 = y4[1];     // x[ +0.. +7]  (low nibbles,  sub-block 2*iq)
                const float4 yl2 = y4[8],  yl3 = y4[9];     // x[+32..+39]  (high nibbles, sub-block 2*iq+1)
                const float4 yh0 = y4[32], yh1 = y4[33];    // x[+128..]    (low nibbles,  sub-block 2*iq+4)
                const float4 yh2 = y4[40], yh3 = y4[41];    // x[+160..]    (high nibbles, sub-block 2*iq+5)

                const float s0 = lr_sum4(yl0 + yl1);
                const float s1 = lr_sum4(yl2 + yl3);
                const float s2 = lr_sum4(yh0 + yh1);
                const float s3 = lr_sum4(yh2 + yh3);

                FOR_UNROLL (short row = 0; row < LR_NR0; ++row) {
                    device const block_q4_K * blk = (device const block_q4_K *)(brow + (ulong) row * args.nb01) + ib;

                    device const uchar4 * q1 = (device const uchar4 *)(blk->qs + 32 * iq + 8 * ir);
                    device const uchar4 * q2 = q1 + 16;                                 // +64 bytes

                    const float d    = (float) blk->d;
                    const float dmin = (float) blk->dmin;

                    const uchar2 sm0 = get_scale_min_k4_just2(0, 2 * iq + 0, blk->scales);  // sub-block 2*iq
                    const uchar2 sm1 = get_scale_min_k4_just2(0, 2 * iq + 1, blk->scales);  // sub-block 2*iq+1
                    const uchar2 sm2 = get_scale_min_k4_just2(4, 2 * iq + 0, blk->scales);  // sub-block 2*iq+4
                    const uchar2 sm3 = get_scale_min_k4_just2(4, 2 * iq + 1, blk->scales);  // sub-block 2*iq+5

                    const uchar4 qa = q1[0], qb = q1[1], qc = q2[0], qd = q2[1];

                    float4 acc = fma(float4(qa & m0F), yl0, float4(qb & m0F) * yl1) * float(sm0.x);
                    acc = fma(fma(float4(qa & mF0), yl2, float4(qb & mF0) * yl3), float4(float(sm1.x) * (1.f/16.f)), acc);
                    acc = fma(fma(float4(qc & m0F), yh0, float4(qd & m0F) * yh1), float4(float(sm2.x)), acc);
                    acc = fma(fma(float4(qc & mF0), yh2, float4(qd & mF0) * yh3), float4(float(sm3.x) * (1.f/16.f)), acc);

                    sumf[row] += d * lr_sum4(acc)
                               - dmin * (s0 * float(sm0.y) + s1 * float(sm1.y) + s2 * float(sm2.y) + s3 * float(sm3.y));
                }
            }

            FOR_UNROLL (short row = 0; row < LR_NR0; ++row) {
                const float s = simd_sum(sumf[row]);
                if (tiisg == 0) {
                    lds_Z[lr_zidx(r0 + row)] = s;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------------------------------------
    // Phase 2: y[row] = A[row, :] . Z      (one block_q4_K per row)
    // lane -> (slot, j): slot = row slot (4 rows per simdgroup), j = 16-byte slice of the 128-byte qs
    //   j = 2*g + h : group g (64 values), half h  ->  low nibbles  = Z[64g + 16h + 0..15]
    //                                                  high nibbles = Z[64g + 32 + 16h + 0..15]
    // ------------------------------------------------------------------------------------------------
    {
        const short slot = tiisg / 8;
        const short j    = tiisg % 8;
        const short g    = j >> 1;
        const short h    = j & 1;

        // Z slice for this lane: one read from threadgroup memory, then registers for the whole row loop
        float4 zl[4];
        float4 zh[4];
        const int q0 = 16 * g + 4 * h;
        FOR_UNROLL (short i = 0; i < 4; ++i) {
            zl[i] = lds_Z4[lr_swz(q0 + i)];
            zh[i] = lds_Z4[lr_swz(q0 + 8 + i)];
        }
        const float sL = lr_sum4((zl[0] + zl[1]) + (zl[2] + zl[3]));
        const float sH = lr_sum4((zh[0] + zh[1]) + (zh[2] + zh[3]));

        const int ntg     = (int) tgpg.x;
        const int ngroups = (args.ne0 + 15) / 16;

        for (int gi = (int) tgpig.x; gi < ngroups; gi += LR_RB * ntg) {
            int                        row[LR_RB];
            device const block_q4_K *  blk[LR_RB];
            uint4                      w  [LR_RB];

            // issue all loads of this iteration first
            FOR_UNROLL (short u = 0; u < LR_RB; ++u) {
                row[u] = (gi + u * ntg) * 16 + sgitg * 4 + slot;
                const int rowc = min(row[u], args.ne0 - 1);             // keep addresses in range, result is discarded
                blk[u] = (device const block_q4_K *)(srcA + (ulong) rowc * args.nb1A);
                w[u]   = ((device const uint4 *) blk[u]->qs)[j];
            }

            FOR_UNROLL (short u = 0; u < LR_RB; ++u) {
                const float d    = (float) blk[u]->d;
                const float dmin = (float) blk[u]->dmin;

                const uchar2 smL = get_scale_min_k4_just2(2 * g, 0, blk[u]->scales);    // sub-block 2g
                const uchar2 smH = get_scale_min_k4_just2(2 * g, 1, blk[u]->scales);    // sub-block 2g+1

                const uint4 ww = w[u];

                float4 aL = float4(as_type<uchar4>(ww.x & 0x0F0F0F0Fu)) * zl[0];
                aL = fma(float4(as_type<uchar4>(ww.y & 0x0F0F0F0Fu)), zl[1], aL);
                aL = fma(float4(as_type<uchar4>(ww.z & 0x0F0F0F0Fu)), zl[2], aL);
                aL = fma(float4(as_type<uchar4>(ww.w & 0x0F0F0F0Fu)), zl[3], aL);

                float4 aH = float4(as_type<uchar4>(ww.x & 0xF0F0F0F0u)) * zh[0];
                aH = fma(float4(as_type<uchar4>(ww.y & 0xF0F0F0F0u)), zh[1], aH);
                aH = fma(float4(as_type<uchar4>(ww.z & 0xF0F0F0F0u)), zh[2], aH);
                aH = fma(float4(as_type<uchar4>(ww.w & 0xF0F0F0F0u)), zh[3], aH);

                float v = d * (lr_sum4(aL) * float(smL.x) + lr_sum4(aH) * (float(smH.x) * (1.f/16.f)))
                        - dmin * (float(smL.y) * sL + float(smH.y) * sH);

                v += simd_shuffle_xor(v, 4);
                v += simd_shuffle_xor(v, 2);
                v += simd_shuffle_xor(v, 1);

                if (j == 0 && row[u] < args.ne0) {
                    dst_f[row[u]] = v;
                }
            }
        }
    }
}
