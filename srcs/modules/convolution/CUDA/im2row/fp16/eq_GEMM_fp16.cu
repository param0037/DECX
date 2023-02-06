/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "eq_GEMM_fp16.cuh"


__global__ void 
decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16(float4*                  A,
                        float4*                  B,
                        float4*                  dst,
                        const ulong2             MatA_load_bounds,
                        const uint               dst_store_bound,                // in float4
                        const uint               WB,                             // in float4
                        const uint               __iter)                         // how many iteration along B's width (128x)
{
#if __ABOVE_SM_53
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x,    // [0, 32)
               tidy = threadIdx.y + blockIdx.y * blockDim.y;    // [0, 8)

    uint along_L_A;

    half2_8 reg_A[4][2], reg_B[4][2];
    half2 reg_sum[8][4];
    
    _SET_ZERO_FLOAT4_1x4_(reg_sum);

    __shared__ float4 frag_A[128][8 + 1];
    __shared__ float4 frag_B[64][16 + 1];

    size_t dex_A, dex_B, dex_dst;

    along_L_A = threadIdx.y;
    dex_A = tidx * MatA_load_bounds.x * 4 + along_L_A;
    dex_B = ((blockIdx.y * blockDim.x + threadIdx.x) * WB + threadIdx.y) * 2;

    for (uint i = 0; i < __iter; ++i)
    {
        _SET_ZERO_FLOAT4_2x4_(reg_B);

        *((float4*)&reg_B[0][0]) = B[dex_B];
        *((float4*)&reg_B[1][0]) = B[dex_B + 1];
        *((float4*)&reg_B[2][0]) = B[dex_B + WB];
        *((float4*)&reg_B[3][0]) = B[dex_B + WB + 1];

        frag_B[threadIdx.x * 2][threadIdx.y * 2] = *((float4*)&reg_B[0][0]);
        frag_B[threadIdx.x * 2][threadIdx.y * 2 + 1] = *((float4*)&reg_B[1][0]);
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2] = *((float4*)&reg_B[2][0]);
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = *((float4*)&reg_B[3][0]);

        for (uint load_A = 0; load_A < 2; ++load_A) {
            // square 00
            _SET_ZERO_FLOAT4_2x4_(reg_A);

            if (tidx < MatA_load_bounds.y && along_L_A < MatA_load_bounds.x) {
                *((float4*)&reg_A[0][0]) = A[dex_A];
                *((float4*)&reg_A[1][0]) = A[dex_A + MatA_load_bounds.x];
                *((float4*)&reg_A[2][0]) = A[dex_A + MatA_load_bounds.x * 2];
                *((float4*)&reg_A[3][0]) = A[dex_A + MatA_load_bounds.x * 3];
            }

            frag_A[threadIdx.x * 4][threadIdx.y] = *((float4*)&reg_A[0][0]);
            frag_A[threadIdx.x * 4 + 1][threadIdx.y] = *((float4*)&reg_A[1][0]);
            frag_A[threadIdx.x * 4 + 2][threadIdx.y] = *((float4*)&reg_A[2][0]);
            frag_A[threadIdx.x * 4 + 3][threadIdx.y] = *((float4*)&reg_A[3][0]);

            along_L_A += 8;
            dex_A += 8;

            __syncthreads();

            // calculation
            for (uint _L = 0; _L < 4; ++_L) {
                *((float4*)&reg_A[0][0]) = frag_A[threadIdx.x * 4][_L * 2];
                *((float4*)&reg_A[0][1]) = frag_A[threadIdx.x * 4][_L * 2 + 1];
                *((float4*)&reg_A[1][0]) = frag_A[threadIdx.x * 4 + 1][_L * 2];
                *((float4*)&reg_A[1][1]) = frag_A[threadIdx.x * 4 + 1][_L * 2 + 1];
                *((float4*)&reg_A[2][0]) = frag_A[threadIdx.x * 4 + 2][_L * 2];
                *((float4*)&reg_A[2][1]) = frag_A[threadIdx.x * 4 + 2][_L * 2 + 1];
                *((float4*)&reg_A[3][0]) = frag_A[threadIdx.x * 4 + 3][_L * 2];
                *((float4*)&reg_A[3][1]) = frag_A[threadIdx.x * 4 + 3][_L * 2 + 1];

                *((float4*)&reg_B[0][0]) = frag_B[threadIdx.y * 8][load_A * 8 + _L * 2];
                *((float4*)&reg_B[0][1]) = frag_B[threadIdx.y * 8][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[1][0]) = frag_B[threadIdx.y * 8 + 1][load_A * 8 + _L * 2];
                *((float4*)&reg_B[1][1]) = frag_B[threadIdx.y * 8 + 1][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[2][0]) = frag_B[threadIdx.y * 8 + 2][load_A * 8 + _L * 2];
                *((float4*)&reg_B[2][1]) = frag_B[threadIdx.y * 8 + 2][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[3][0]) = frag_B[threadIdx.y * 8 + 3][load_A * 8 + _L * 2];
                *((float4*)&reg_B[3][1]) = frag_B[threadIdx.y * 8 + 3][load_A * 8 + _L * 2 + 1];
                
                decx::conv::GPUK::_MM_4x4x8_fp16<0>(reg_A, reg_B, reg_sum);

                *((float4*)&reg_B[0][0]) = frag_B[threadIdx.y * 8 + 4][load_A * 8 + _L * 2];
                *((float4*)&reg_B[0][1]) = frag_B[threadIdx.y * 8 + 4][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[1][0]) = frag_B[threadIdx.y * 8 + 5][load_A * 8 + _L * 2];
                *((float4*)&reg_B[1][1]) = frag_B[threadIdx.y * 8 + 5][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[2][0]) = frag_B[threadIdx.y * 8 + 6][load_A * 8 + _L * 2];
                *((float4*)&reg_B[2][1]) = frag_B[threadIdx.y * 8 + 6][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[3][0]) = frag_B[threadIdx.y * 8 + 7][load_A * 8 + _L * 2];
                *((float4*)&reg_B[3][1]) = frag_B[threadIdx.y * 8 + 7][load_A * 8 + _L * 2 + 1];

                decx::conv::GPUK::_MM_4x4x8_fp16<4>(reg_A, reg_B, reg_sum);
            }

            __syncthreads();
        }

        dex_B += 16;
    }

    dex_A = tidx * dst_store_bound * 4 + tidy;

    for (int i = 0; i < 4; ++i) {
        reg_A[i][0].x.x = __hadd(reg_sum[i][0].x, reg_sum[i][0].y);
        reg_A[i][0].x.y = __hadd(reg_sum[i][1].x, reg_sum[i][1].y);
        reg_A[i][0].y.x = __hadd(reg_sum[i][2].x, reg_sum[i][2].y);
        reg_A[i][0].y.y = __hadd(reg_sum[i][3].x, reg_sum[i][3].y);
        reg_A[i][0].z.x = __hadd(reg_sum[i + 4][0].x, reg_sum[i + 4][0].y);
        reg_A[i][0].z.y = __hadd(reg_sum[i + 4][1].x, reg_sum[i + 4][1].y);
        reg_A[i][0].w.x = __hadd(reg_sum[i + 4][2].x, reg_sum[i + 4][2].y);
        reg_A[i][0].w.y = __hadd(reg_sum[i + 4][3].x, reg_sum[i + 4][3].y);
    }

    if (tidy < dst_store_bound) {
        _STORE_TO_DEST_FP16_(0);
        _STORE_TO_DEST_FP16_(1);
        _STORE_TO_DEST_FP16_(2);
        _STORE_TO_DEST_FP16_(3);
    }
#endif
}


__global__ void 
decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16_accu(float4*                  A,
                             float4*                  B,
                             float4*                  dst,
                             const ulong2             MatA_load_bounds,
                             const uint               dst_store_bound,                // in float4
                             const uint               WB,                             // in float4
                             const uint               __iter)                         // how many iteration along B's width (128x)
{
#if __ABOVE_SM_53
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x,    // [0, 32)
               tidy = threadIdx.y + blockIdx.y * blockDim.y;    // [0, 8)

    uint along_L_A;

    half2_8 reg_A[4][2], reg_B[4][2];
    float reg_sum[8][4];
    
    _SET_ZERO_FLOAT4_1x4_(reg_sum);

    __shared__ float4 frag_A[128][8 + 1];
    __shared__ float4 frag_B[64][16 + 1];

    size_t dex_A, dex_B, dex_dst;

    along_L_A = threadIdx.y;
    dex_A = tidx * MatA_load_bounds.x * 4 + along_L_A;
    dex_B = ((blockIdx.y * blockDim.x + threadIdx.x) * WB + threadIdx.y) * 2;

    for (uint i = 0; i < __iter; ++i)
    {
        _SET_ZERO_FLOAT4_2x4_(reg_B);

        *((float4*)&reg_B[0][0]) = B[dex_B];
        *((float4*)&reg_B[1][0]) = B[dex_B + 1];
        *((float4*)&reg_B[2][0]) = B[dex_B + WB];
        *((float4*)&reg_B[3][0]) = B[dex_B + WB + 1];

        frag_B[threadIdx.x * 2][threadIdx.y * 2] = *((float4*)&reg_B[0][0]);
        frag_B[threadIdx.x * 2][threadIdx.y * 2 + 1] = *((float4*)&reg_B[1][0]);
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2] = *((float4*)&reg_B[2][0]);
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = *((float4*)&reg_B[3][0]);

        for (uint load_A = 0; load_A < 2; ++load_A) {
            // square 00
            _SET_ZERO_FLOAT4_2x4_(reg_A);

            if (tidx < MatA_load_bounds.y && along_L_A < MatA_load_bounds.x) {
                *((float4*)&reg_A[0][0]) = A[dex_A];
                *((float4*)&reg_A[1][0]) = A[dex_A + MatA_load_bounds.x];
                *((float4*)&reg_A[2][0]) = A[dex_A + MatA_load_bounds.x * 2];
                *((float4*)&reg_A[3][0]) = A[dex_A + MatA_load_bounds.x * 3];
            }

            frag_A[threadIdx.x * 4][threadIdx.y] = *((float4*)&reg_A[0][0]);
            frag_A[threadIdx.x * 4 + 1][threadIdx.y] = *((float4*)&reg_A[1][0]);
            frag_A[threadIdx.x * 4 + 2][threadIdx.y] = *((float4*)&reg_A[2][0]);
            frag_A[threadIdx.x * 4 + 3][threadIdx.y] = *((float4*)&reg_A[3][0]);

            along_L_A += 8;
            dex_A += 8;

            __syncthreads();

            // calculation
            for (uint _L = 0; _L < 4; ++_L) {
                *((float4*)&reg_A[0][0]) = frag_A[threadIdx.x * 4][_L * 2];
                *((float4*)&reg_A[0][1]) = frag_A[threadIdx.x * 4][_L * 2 + 1];
                *((float4*)&reg_A[1][0]) = frag_A[threadIdx.x * 4 + 1][_L * 2];
                *((float4*)&reg_A[1][1]) = frag_A[threadIdx.x * 4 + 1][_L * 2 + 1];
                *((float4*)&reg_A[2][0]) = frag_A[threadIdx.x * 4 + 2][_L * 2];
                *((float4*)&reg_A[2][1]) = frag_A[threadIdx.x * 4 + 2][_L * 2 + 1];
                *((float4*)&reg_A[3][0]) = frag_A[threadIdx.x * 4 + 3][_L * 2];
                *((float4*)&reg_A[3][1]) = frag_A[threadIdx.x * 4 + 3][_L * 2 + 1];

                *((float4*)&reg_B[0][0]) = frag_B[threadIdx.y * 8][load_A * 8 + _L * 2];
                *((float4*)&reg_B[0][1]) = frag_B[threadIdx.y * 8][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[1][0]) = frag_B[threadIdx.y * 8 + 1][load_A * 8 + _L * 2];
                *((float4*)&reg_B[1][1]) = frag_B[threadIdx.y * 8 + 1][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[2][0]) = frag_B[threadIdx.y * 8 + 2][load_A * 8 + _L * 2];
                *((float4*)&reg_B[2][1]) = frag_B[threadIdx.y * 8 + 2][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[3][0]) = frag_B[threadIdx.y * 8 + 3][load_A * 8 + _L * 2];
                *((float4*)&reg_B[3][1]) = frag_B[threadIdx.y * 8 + 3][load_A * 8 + _L * 2 + 1];
                
                decx::conv::GPUK::_MM_4x4x8_fp16_accu<0>(reg_A, reg_B, reg_sum);

                *((float4*)&reg_B[0][0]) = frag_B[threadIdx.y * 8 + 4][load_A * 8 + _L * 2];
                *((float4*)&reg_B[0][1]) = frag_B[threadIdx.y * 8 + 4][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[1][0]) = frag_B[threadIdx.y * 8 + 5][load_A * 8 + _L * 2];
                *((float4*)&reg_B[1][1]) = frag_B[threadIdx.y * 8 + 5][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[2][0]) = frag_B[threadIdx.y * 8 + 6][load_A * 8 + _L * 2];
                *((float4*)&reg_B[2][1]) = frag_B[threadIdx.y * 8 + 6][load_A * 8 + _L * 2 + 1];
                *((float4*)&reg_B[3][0]) = frag_B[threadIdx.y * 8 + 7][load_A * 8 + _L * 2];
                *((float4*)&reg_B[3][1]) = frag_B[threadIdx.y * 8 + 7][load_A * 8 + _L * 2 + 1];

                decx::conv::GPUK::_MM_4x4x8_fp16_accu<4>(reg_A, reg_B, reg_sum);
            }

            __syncthreads();
        }

        dex_B += 16;
    }

    dex_A = tidx * dst_store_bound * 4 + tidy;

    for (int i = 0; i < 4; ++i) {
        reg_A[i][0].x.x = __float2half(reg_sum[i][0]);
        reg_A[i][0].x.y = __float2half(reg_sum[i][1]);
        reg_A[i][0].y.x = __float2half(reg_sum[i][2]);
        reg_A[i][0].y.y = __float2half(reg_sum[i][3]);
        reg_A[i][0].z.x = __float2half(reg_sum[i + 4][0]);
        reg_A[i][0].z.y = __float2half(reg_sum[i + 4][1]);
        reg_A[i][0].w.x = __float2half(reg_sum[i + 4][2]);
        reg_A[i][0].w.y = __float2half(reg_sum[i + 4][3]);
    }

    if (tidy < dst_store_bound) {
        _STORE_TO_DEST_FP16_(0);
        _STORE_TO_DEST_FP16_(1);
        _STORE_TO_DEST_FP16_(2);
        _STORE_TO_DEST_FP16_(3);
    }
#endif
}