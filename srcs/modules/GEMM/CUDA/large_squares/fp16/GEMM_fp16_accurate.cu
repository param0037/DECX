/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "GEMM_fp16_accurate.cuh"


__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_spec_accu(float4 *                A,
                            float4 *                B,
                            float4 *                dst,
                            const uint              pitch_A,
                            const uint              pitch_B,
                            const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    float4 sum[8][2];
    
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
        *((float4*)&sum[i][1]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    
    half reg_0[8];        // the total size is equal to a float4
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    size_t glo_dex_A = (threadIdx.x / 2 + 128 * blockIdx.x) * pitch_A + threadIdx.x % 2;
    size_t glo_dex_B = ((threadIdx.x % 16) + blockIdx.y * 16) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = A[glo_dex_A];
        
        x_glo = 8 * (threadIdx.x % 2);            y_glo = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = reg_0[0];
        *((half*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = reg_0[1];
        *((half*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = reg_0[2];
        *((half*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = reg_0[3];
        *((half*)&(shmemA[x_glo + 4][threadIdx.x / 16]) + y_glo) = reg_0[4];
        *((half*)&(shmemA[x_glo + 5][threadIdx.x / 16]) + y_glo) = reg_0[5];
        *((half*)&(shmemA[x_glo + 6][threadIdx.x / 16]) + y_glo) = reg_0[6];
        *((half*)&(shmemA[x_glo + 7][threadIdx.x / 16]) + y_glo) = reg_0[7];
        
        x_glo = threadIdx.x / 16;                y_glo = threadIdx.x % 16;

        shmemB[x_glo][y_glo] = B[glo_dex_B];            //load globalB to shmemB
        
        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_glo];
            *((float4*)&reg_1) = shmemB[__line][y_glo];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo;

    h_cvt_and_store;

#endif
}


// last storage (16, 16)
// ¼ÆËã / ·Ã´æ ±È is the crucial, reduce memory assess by vectorization
__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_spec_accu(float4 *                A,
                                float4 *                B,
                                float4 *                C,
                                float4 *                dst,
                                const uint              pitch_A,
                                const uint              pitch_B,
                                const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_glo, y_glo;
    size_t glo_dex_A, glo_dex_B;
    
    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    float4 sum[8][2];
    half reg_0[8];        // the total size is equal to a float4
    half2 reg_1[4], mul_tmp0, mul_tmp1;
    
    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo;

    h_cvt_loadC;        // load from matrix C

    glo_dex_A = (threadIdx.x / 2 + 128 * blockIdx.x) * pitch_A + threadIdx.x % 2;
    glo_dex_B = ((threadIdx.x % 16) + blockIdx.y * 16) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = A[glo_dex_A];
        
        x_glo = 8 * (threadIdx.x % 2);            y_glo = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = reg_0[0];
        *((half*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = reg_0[1];
        *((half*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = reg_0[2];
        *((half*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = reg_0[3];
        *((half*)&(shmemA[x_glo + 4][threadIdx.x / 16]) + y_glo) = reg_0[4];
        *((half*)&(shmemA[x_glo + 5][threadIdx.x / 16]) + y_glo) = reg_0[5];
        *((half*)&(shmemA[x_glo + 6][threadIdx.x / 16]) + y_glo) = reg_0[6];
        *((half*)&(shmemA[x_glo + 7][threadIdx.x / 16]) + y_glo) = reg_0[7];
        
        x_glo = threadIdx.x / 16;                y_glo = threadIdx.x % 16;

        shmemB[x_glo][y_glo] = B[glo_dex_B];            //load globalB to shmemB
        
        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_glo];
            *((float4*)&reg_1) = shmemB[__line][y_glo];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo;

    h_cvt_and_store;

#endif
}



__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_anyWH_anyL_accu(float4 *                A,
                                  float4 *                B,
                                  float4 *                dst,
                                  const uint              pitch_A,
                                  const uint              pitch_B,
                                  const uint              Hdst,
                                  const uint              HB,
                                  const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    float4 sum[8][2];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
        *((float4*)&sum[i][1]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half reg_0[8];
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst && y_gloA < pitch_A)
            *((float4*)&reg_0) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = reg_0[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = reg_0[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = reg_0[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = reg_0[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = reg_0[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = reg_0[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = reg_0[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = reg_0[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B && x_gloB < HB)
            *((float4*)&reg_0) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&reg_0);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_loc];
            *((float4*)&reg_1) = shmemB[__line][y_loc];

            sfma_8x8_mix_accu;
}
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_cvt_and_store_a_line(0);
        if (x_gloA + 1 < Hdst)      h_cvt_and_store_a_line(1);
        if (x_gloA + 2 < Hdst)      h_cvt_and_store_a_line(2);
        if (x_gloA + 3 < Hdst)      h_cvt_and_store_a_line(3);
        if (x_gloA + 4 < Hdst)      h_cvt_and_store_a_line(4);
        if (x_gloA + 5 < Hdst)      h_cvt_and_store_a_line(5);
        if (x_gloA + 6 < Hdst)      h_cvt_and_store_a_line(6);
        if (x_gloA + 7 < Hdst)      h_cvt_and_store_a_line(7);
    }
#endif
}



__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_anyWH_anyL_accu(float4 *                A,
                                      float4 *                B,
                                      float4 *                C,
                                      float4 *                dst,
                                      const uint              pitch_A,
                                      const uint              pitch_B,
                                      const uint              Hdst,
                                      const uint              HB,
                                      const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    float4 sum[8][2];
    half reg_0[8];
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)           h_cvt_loadC_a_line(0);
        if (x_gloA + 1 < Hdst)       h_cvt_loadC_a_line(1);
        if (x_gloA + 2 < Hdst)       h_cvt_loadC_a_line(2);
        if (x_gloA + 3 < Hdst)       h_cvt_loadC_a_line(3);
        if (x_gloA + 4 < Hdst)       h_cvt_loadC_a_line(4);
        if (x_gloA + 5 < Hdst)       h_cvt_loadC_a_line(5);
        if (x_gloA + 6 < Hdst)       h_cvt_loadC_a_line(6);
        if (x_gloA + 7 < Hdst)       h_cvt_loadC_a_line(7);
    }

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst && y_gloA < pitch_A)
            *((float4*)&reg_0) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = reg_0[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = reg_0[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = reg_0[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = reg_0[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = reg_0[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = reg_0[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = reg_0[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = reg_0[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B && x_gloB < HB)
            *((float4*)&reg_0) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&reg_0);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_loc];
            *((float4*)&reg_1) = shmemB[__line][y_loc];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_cvt_and_store_a_line(0);
        if (x_gloA + 1 < Hdst)      h_cvt_and_store_a_line(1);
        if (x_gloA + 2 < Hdst)      h_cvt_and_store_a_line(2);
        if (x_gloA + 3 < Hdst)      h_cvt_and_store_a_line(3);
        if (x_gloA + 4 < Hdst)      h_cvt_and_store_a_line(4);
        if (x_gloA + 5 < Hdst)      h_cvt_and_store_a_line(5);
        if (x_gloA + 6 < Hdst)      h_cvt_and_store_a_line(6);
        if (x_gloA + 7 < Hdst)      h_cvt_and_store_a_line(7);
    }
#endif
}



__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_specWH_anyL_accu(float4 *                A,
                                   float4 *                B,
                                   float4 *                dst,
                                   const uint              pitch_A,
                                   const uint              pitch_B,
                                   const uint              HB,
                                   const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    float4 sum[8][2];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
        *((float4*)&sum[i][1]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half reg_0[8];
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A)
            *((float4*)&reg_0) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = reg_0[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = reg_0[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = reg_0[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = reg_0[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = reg_0[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = reg_0[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = reg_0[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = reg_0[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (x_gloB < HB)
            *((float4*)&reg_0) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&reg_0);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_loc];
            *((float4*)&reg_1) = shmemB[__line][y_loc];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    h_cvt_and_store;
#endif
}




__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_specWH_anyL_accu(float4 *                A,
                                       float4 *                B,
                                       float4 *                C,
                                       float4 *                dst,
                                       const uint              pitch_A,
                                       const uint              pitch_B,
                                       const uint              HB,
                                       const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    float4 sum[8][2];
    half reg_0[8];
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    h_cvt_loadC;            // load from matrix C

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A)
            *((float4*)&reg_0) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = reg_0[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = reg_0[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = reg_0[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = reg_0[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = reg_0[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = reg_0[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = reg_0[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = reg_0[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (x_gloB < HB)
            *((float4*)&reg_0) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&reg_0);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_loc];
            *((float4*)&reg_1) = shmemB[__line][y_loc];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    h_cvt_and_store;
#endif
}



__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_anyWH_specL_accu(float4 *                A,
                                   float4 *                B,
                                   float4 *                dst,
                                   const uint              pitch_A,
                                   const uint              pitch_B,
                                   const uint              Hdst,
                                   const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    float4 sum[8][2];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
        *((float4*)&sum[i][1]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half reg_0[8];
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            *((float4*)&reg_0) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = reg_0[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = reg_0[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = reg_0[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = reg_0[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = reg_0[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = reg_0[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = reg_0[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = reg_0[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            *((float4*)&reg_0) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&reg_0);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_loc];
            *((float4*)&reg_1) = shmemB[__line][y_loc];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_cvt_and_store_a_line(0);
        if (x_gloA + 1 < Hdst)      h_cvt_and_store_a_line(1);
        if (x_gloA + 2 < Hdst)      h_cvt_and_store_a_line(2);
        if (x_gloA + 3 < Hdst)      h_cvt_and_store_a_line(3);
        if (x_gloA + 4 < Hdst)      h_cvt_and_store_a_line(4);
        if (x_gloA + 5 < Hdst)      h_cvt_and_store_a_line(5);
        if (x_gloA + 6 < Hdst)      h_cvt_and_store_a_line(6);
        if (x_gloA + 7 < Hdst)      h_cvt_and_store_a_line(7);
    }
#endif
}



__global__
/**
* config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
* __same should be 16-times and dstDims should be both 128-times
* @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
* @param pitch_dst : considered float4
* @param __iter : __linear(in float) / 16
*/
void cu_GEMM_fp16_ABC_anyWH_specL_accu(float4 *                A,
                                       float4 *                B,
                                       float4 *                C,
                                       float4 *                dst,
                                       const uint              pitch_A,
                                       const uint              pitch_B,
                                       const uint              Hdst,
                                       const uint              __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    float4 sum[8][2];
    half reg_0[8];
    half2 reg_1[4], mul_tmp0, mul_tmp1;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)           h_cvt_loadC_a_line(0);
        if (x_gloA + 1 < Hdst)       h_cvt_loadC_a_line(1);
        if (x_gloA + 2 < Hdst)       h_cvt_loadC_a_line(2);
        if (x_gloA + 3 < Hdst)       h_cvt_loadC_a_line(3);
        if (x_gloA + 4 < Hdst)       h_cvt_loadC_a_line(4);
        if (x_gloA + 5 < Hdst)       h_cvt_loadC_a_line(5);
        if (x_gloA + 6 < Hdst)       h_cvt_loadC_a_line(6);
        if (x_gloA + 7 < Hdst)       h_cvt_loadC_a_line(7);
    }

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&reg_0) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            *((float4*)&reg_0) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = reg_0[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = reg_0[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = reg_0[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = reg_0[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = reg_0[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = reg_0[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = reg_0[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = reg_0[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            *((float4*)&reg_0) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&reg_0);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&reg_0) = shmemA[__line][x_loc];
            *((float4*)&reg_1) = shmemB[__line][y_loc];

            sfma_8x8_mix_accu;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_cvt_and_store_a_line(0);
        if (x_gloA + 1 < Hdst)      h_cvt_and_store_a_line(1);
        if (x_gloA + 2 < Hdst)      h_cvt_and_store_a_line(2);
        if (x_gloA + 3 < Hdst)      h_cvt_and_store_a_line(3);
        if (x_gloA + 4 < Hdst)      h_cvt_and_store_a_line(4);
        if (x_gloA + 5 < Hdst)      h_cvt_and_store_a_line(5);
        if (x_gloA + 6 < Hdst)      h_cvt_and_store_a_line(6);
        if (x_gloA + 7 < Hdst)      h_cvt_and_store_a_line(7);
    }
#endif
}
