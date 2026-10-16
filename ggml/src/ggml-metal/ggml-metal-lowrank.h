// ggml-metal-lowrank.h
//
// Host/device definitions shared by ggml-metal-ops.cpp and kernels/mul_mv_lowrank_fused.metal
// (Latent-DFlash stage 3: fused low-rank FFN projection  y = A . (B . x),  Q4_K factors, rank 256).
#pragma once

#ifndef __METAL_VERSION__
#include <stdint.h>
#endif

#define LOWRANK_FUSED_RANK      256     // B has 256 rows, A has 256 columns (exactly one block_q4_K per A row)
#define LOWRANK_FUSED_NTHREADS  128     // 4 simdgroups
#define LOWRANK_FUSED_SMEM      1024    // Z[256] in FP32, threadgroup memory
#define LOWRANK_FUSED_MAX_COLS  8       // above this the generic mul_mm path is the better choice

typedef struct {
    int32_t  ne00;      // D_in  : length of x == elements per Factor-B row (multiple of 256)
    int32_t  ne0;       // D_out : rows of Factor A == length of y
    int32_t  ne11;      // number of activation columns (grid.y)
    int32_t  _pad;
    uint64_t nb01;      // Factor B row stride, bytes
    uint64_t nb1A;      // Factor A row stride, bytes (144)
    uint64_t nb11;      // x column stride, bytes
    uint64_t nb1;       // y column stride, bytes
} ggml_metal_kargs_mul_mv_lowrank_fused;
