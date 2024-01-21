/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "eq_GEMM_fp16.cuh"


__global__ void 
decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16(const float4* __restrict       A,
                                         const float4* __restrict       B,
                                         float4*                        dst,
                                         const ulong2                   MatA_load_bounds,
                                         const uint                     dst_store_bound,                // in float4
                                         const uint                     WB,                             // in float4
                                         const uint                     __iter)                         // how many iteration along B's width (128x)
{
#if __ABOVE_SM_53
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x,    // [0, 32), adj
               tidy = threadIdx.y + blockIdx.y * blockDim.y;    // [0, 8), stride

    uint along_L_A;

    decx::utils::_cuda_vec128 reg_A[8], reg_B[8];
    __half2 reg_sum[8][8];
    
    _SET_ZERO_FLOAT4_1x8_(reg_sum);

    __shared__ float4 frag_A[32][32 + 1];
    __shared__ float4 frag_B[64][16 + 1];

    size_t dex_A, dex_B, dex_dst;

    along_L_A = threadIdx.y;
    dex_A = along_L_A * MatA_load_bounds.x * 4 + tidx;
    dex_B = ((blockIdx.y * blockDim.x + threadIdx.x) * WB + threadIdx.y) * 2;

    for (uint i = 0; i < __iter; ++i)
    {
        _SET_ZERO_FLOAT4_1x8_(reg_B);

        reg_B[0]._vf = B[dex_B];
        reg_B[1]._vf = B[dex_B + 1];
        reg_B[2]._vf = B[dex_B + WB];
        reg_B[3]._vf = B[dex_B + WB + 1];

        frag_B[threadIdx.x * 2][threadIdx.y * 2] = reg_B[0]._vf;
        frag_B[threadIdx.x * 2][threadIdx.y * 2 + 1] = reg_B[1]._vf;
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2] = reg_B[2]._vf;
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = reg_B[3]._vf;

        for (uint load_A = 0; load_A < 4; ++load_A) {
            // square 00
            _SET_ZERO_FLOAT4_1x8_(reg_A);

            if (tidx < MatA_load_bounds.x && along_L_A < MatA_load_bounds.y) {
                reg_A[0]._vf = A[dex_A];
                reg_A[1]._vf = A[dex_A + MatA_load_bounds.x];
                reg_A[2]._vf = A[dex_A + MatA_load_bounds.x * 2];
                reg_A[3]._vf = A[dex_A + MatA_load_bounds.x * 3];
            }

            frag_A[threadIdx.y * 4][threadIdx.x] = reg_A[0]._vf;
            frag_A[threadIdx.y * 4 + 1][threadIdx.x] = reg_A[1]._vf;
            frag_A[threadIdx.y * 4 + 2][threadIdx.x] = reg_A[2]._vf;
            frag_A[threadIdx.y * 4 + 3][threadIdx.x] = reg_A[3]._vf;

            along_L_A += 8;
            dex_A += 8 * 4 * MatA_load_bounds.x;

            __syncthreads();

            // calculation
            for (uint _L = 0; _L < 4; ++_L) 
            {
                reg_A[0]._vf = frag_A[_L * 8][threadIdx.x];
                reg_A[1]._vf = frag_A[_L * 8 + 1][threadIdx.x];
                reg_A[2]._vf = frag_A[_L * 8 + 2][threadIdx.x];
                reg_A[3]._vf = frag_A[_L * 8 + 3][threadIdx.x];
                reg_A[4]._vf = frag_A[_L * 8 + 4][threadIdx.x];
                reg_A[5]._vf = frag_A[_L * 8 + 5][threadIdx.x];
                reg_A[6]._vf = frag_A[_L * 8 + 6][threadIdx.x];
                reg_A[7]._vf = frag_A[_L * 8 + 7][threadIdx.x];

                reg_B[0]._vf = frag_B[threadIdx.y * 8][load_A * 4 + _L];
                reg_B[1]._vf = frag_B[threadIdx.y * 8 + 1][load_A * 4 + _L];
                reg_B[2]._vf = frag_B[threadIdx.y * 8 + 2][load_A * 4 + _L];
                reg_B[3]._vf = frag_B[threadIdx.y * 8 + 3][load_A * 4 + _L];
                reg_B[4]._vf = frag_B[threadIdx.y * 8 + 4][load_A * 4 + _L];
                reg_B[5]._vf = frag_B[threadIdx.y * 8 + 5][load_A * 4 + _L];
                reg_B[6]._vf = frag_B[threadIdx.y * 8 + 6][load_A * 4 + _L];
                reg_B[7]._vf = frag_B[threadIdx.y * 8 + 7][load_A * 4 + _L];

                decx::conv::GPUK::_MM_8x8x8_fp16(reg_A, reg_B, reg_sum);
            }

            __syncthreads();
        }

        dex_B += 16;
    }

    dex_A = tidx * dst_store_bound * 8 + tidy;

    for (int i = 0; i < 8; ++i) {
        reg_A[i]._arrh[0] = __hadd(reg_sum[i][0].x, reg_sum[i][0].y);
        reg_A[i]._arrh[1] = __hadd(reg_sum[i][1].x, reg_sum[i][1].y);
        reg_A[i]._arrh[2] = __hadd(reg_sum[i][2].x, reg_sum[i][2].y);
        reg_A[i]._arrh[3] = __hadd(reg_sum[i][3].x, reg_sum[i][3].y);
        reg_A[i]._arrh[4] = __hadd(reg_sum[i][4].x, reg_sum[i][4].y);
        reg_A[i]._arrh[5] = __hadd(reg_sum[i][5].x, reg_sum[i][5].y);
        reg_A[i]._arrh[6] = __hadd(reg_sum[i][6].x, reg_sum[i][6].y);
        reg_A[i]._arrh[7] = __hadd(reg_sum[i][7].x, reg_sum[i][7].y);
    }

    if (tidy < dst_store_bound) {
        dst[dex_A] = reg_A[0]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[1]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[2]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[3]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[4]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[5]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[6]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[7]._vf;
    }
#endif
}


__global__ void 
decx::conv_I2R::GPUK::cu_conv_eq_mm_fp16_accu(const float4* __restrict      A,
                                              const float4* __restrict      B,
                                              float4*                       dst,
                                              const ulong2                  MatA_load_bounds,
                                              const uint                    dst_store_bound,                // in float4
                                              const uint                    WB,                             // in float4
                                              const uint                    __iter)                         // how many iteration along B's width (128x)
{
#if __ABOVE_SM_53
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x,    // [0, 32), adj
        tidy = threadIdx.y + blockIdx.y * blockDim.y;    // [0, 8), stride

    uint along_L_A;

    decx::utils::_cuda_vec128 reg_A[8], reg_B[8];
    float reg_sum[8][8];

    for (int i = 0; i < 8; ++i) {
        *((float4*)&reg_sum[i][0]) = decx::utils::vec4_set1_fp32(0);
        *((float4*)&reg_sum[i][4]) = decx::utils::vec4_set1_fp32(0);
    }

    __shared__ float4 frag_A[32][32 + 1];
    __shared__ float4 frag_B[64][16 + 1];

    size_t dex_A, dex_B, dex_dst;

    along_L_A = threadIdx.y;
    dex_A = along_L_A * MatA_load_bounds.x * 4 + tidx;
    dex_B = ((blockIdx.y * blockDim.x + threadIdx.x) * WB + threadIdx.y) * 2;

    for (uint i = 0; i < __iter; ++i)
    {
        _SET_ZERO_FLOAT4_1x8_(reg_B);

        reg_B[0]._vf = B[dex_B];
        reg_B[1]._vf = B[dex_B + 1];
        reg_B[2]._vf = B[dex_B + WB];
        reg_B[3]._vf = B[dex_B + WB + 1];

        frag_B[threadIdx.x * 2][threadIdx.y * 2] = reg_B[0]._vf;
        frag_B[threadIdx.x * 2][threadIdx.y * 2 + 1] = reg_B[1]._vf;
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2] = reg_B[2]._vf;
        frag_B[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1] = reg_B[3]._vf;

        for (uint load_A = 0; load_A < 4; ++load_A) {
            // square 00
            _SET_ZERO_FLOAT4_1x8_(reg_A);

            if (tidx < MatA_load_bounds.x && along_L_A < MatA_load_bounds.y) {
                reg_A[0]._vf = A[dex_A];
                reg_A[1]._vf = A[dex_A + MatA_load_bounds.x];
                reg_A[2]._vf = A[dex_A + MatA_load_bounds.x * 2];
                reg_A[3]._vf = A[dex_A + MatA_load_bounds.x * 3];
            }

            frag_A[threadIdx.y * 4][threadIdx.x] = reg_A[0]._vf;
            frag_A[threadIdx.y * 4 + 1][threadIdx.x] = reg_A[1]._vf;
            frag_A[threadIdx.y * 4 + 2][threadIdx.x] = reg_A[2]._vf;
            frag_A[threadIdx.y * 4 + 3][threadIdx.x] = reg_A[3]._vf;

            along_L_A += 8;
            dex_A += 8 * 4 * MatA_load_bounds.x;

            __syncthreads();

            // calculation
            for (uint _L = 0; _L < 4; ++_L)
            {
                reg_A[0]._vf = frag_A[_L * 8][threadIdx.x];
                reg_A[1]._vf = frag_A[_L * 8 + 1][threadIdx.x];
                reg_A[2]._vf = frag_A[_L * 8 + 2][threadIdx.x];
                reg_A[3]._vf = frag_A[_L * 8 + 3][threadIdx.x];
                reg_A[4]._vf = frag_A[_L * 8 + 4][threadIdx.x];
                reg_A[5]._vf = frag_A[_L * 8 + 5][threadIdx.x];
                reg_A[6]._vf = frag_A[_L * 8 + 6][threadIdx.x];
                reg_A[7]._vf = frag_A[_L * 8 + 7][threadIdx.x];

                reg_B[0]._vf = frag_B[threadIdx.y * 8][load_A * 4 + _L];
                reg_B[1]._vf = frag_B[threadIdx.y * 8 + 1][load_A * 4 + _L];
                reg_B[2]._vf = frag_B[threadIdx.y * 8 + 2][load_A * 4 + _L];
                reg_B[3]._vf = frag_B[threadIdx.y * 8 + 3][load_A * 4 + _L];
                reg_B[4]._vf = frag_B[threadIdx.y * 8 + 4][load_A * 4 + _L];
                reg_B[5]._vf = frag_B[threadIdx.y * 8 + 5][load_A * 4 + _L];
                reg_B[6]._vf = frag_B[threadIdx.y * 8 + 6][load_A * 4 + _L];
                reg_B[7]._vf = frag_B[threadIdx.y * 8 + 7][load_A * 4 + _L];

                decx::conv::GPUK::_MM_8x8x8_fp16_accu(reg_A, reg_B, reg_sum);
            }

            __syncthreads();
        }

        dex_B += 16;
    }

    dex_A = tidx * dst_store_bound * 8 + tidy;

    for (int i = 0; i < 8; ++i) {
        reg_A[i]._arrh[0] = __float2half(reg_sum[i][0]);
        reg_A[i]._arrh[1] = __float2half(reg_sum[i][1]);
        reg_A[i]._arrh[2] = __float2half(reg_sum[i][2]);
        reg_A[i]._arrh[3] = __float2half(reg_sum[i][3]);
        reg_A[i]._arrh[4] = __float2half(reg_sum[i][4]);
        reg_A[i]._arrh[5] = __float2half(reg_sum[i][5]);
        reg_A[i]._arrh[6] = __float2half(reg_sum[i][6]);
        reg_A[i]._arrh[7] = __float2half(reg_sum[i][7]);
    }

    if (tidy < dst_store_bound) {
        dst[dex_A] = reg_A[0]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[1]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[2]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[3]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[4]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[5]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[6]._vf;        dex_A += dst_store_bound;
        dst[dex_A] = reg_A[7]._vf;
    }
#endif
}