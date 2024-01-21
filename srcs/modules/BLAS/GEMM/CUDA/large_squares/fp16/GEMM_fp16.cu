/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "GEMM_fp16.cuh"


// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_spec(float4 *                A,
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
    
    half2 sum[8][4];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    size_t glo_dex_A = (threadIdx.x / 2 + 128 * blockIdx.x) * pitch_A + threadIdx.x % 2;
    size_t glo_dex_B = ((threadIdx.x % 16) + blockIdx.y * 16) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = A[glo_dex_A];
        
        x_glo = 8 * (threadIdx.x % 2);            y_glo = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0];
        *((half*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[1];
        *((half*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[2];
        *((half*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[3];
        *((half*)&(shmemA[x_glo + 4][threadIdx.x / 16]) + y_glo) = tmp_A[4];
        *((half*)&(shmemA[x_glo + 5][threadIdx.x / 16]) + y_glo) = tmp_A[5];
        *((half*)&(shmemA[x_glo + 6][threadIdx.x / 16]) + y_glo) = tmp_A[6];
        *((half*)&(shmemA[x_glo + 7][threadIdx.x / 16]) + y_glo) = tmp_A[7];
        
        x_glo = threadIdx.x / 16;                y_glo = threadIdx.x % 16;

        shmemB[x_glo][y_glo] = B[glo_dex_B];            //load globalB to shmemB
        
        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_glo];
            *((float4*)&tmp_B) = shmemB[__line][y_glo];
            
            hfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo;
    h_store(glo_dex_A)
#endif
}



// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_ABC_spec(float4 *            A,
                           float4 *            B,
                           float4 *            C,
                           float4 *            dst,
                           const uint          pitch_A,
                           const uint          pitch_B,
                           const uint          __iter)
{
#if __ABOVE_SM_53
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];
    
    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    size_t glo_dex_A = x_glo * pitch_B + y_glo;

    half2 sum[8][4];
    h_loadC(glo_dex_A)

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    glo_dex_A = (threadIdx.x / 2 + 128 * blockIdx.x) * pitch_A + threadIdx.x % 2;
    size_t glo_dex_B = ((threadIdx.x % 16) + blockIdx.y * 16) + pitch_B * (threadIdx.x / 16);
    
    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = A[glo_dex_A];
        
        x_glo = 8 * (threadIdx.x % 2);            y_glo = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0];
        *((half*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[1];
        *((half*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[2];
        *((half*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[3];
        *((half*)&(shmemA[x_glo + 4][threadIdx.x / 16]) + y_glo) = tmp_A[4];
        *((half*)&(shmemA[x_glo + 5][threadIdx.x / 16]) + y_glo) = tmp_A[5];
        *((half*)&(shmemA[x_glo + 6][threadIdx.x / 16]) + y_glo) = tmp_A[6];
        *((half*)&(shmemA[x_glo + 7][threadIdx.x / 16]) + y_glo) = tmp_A[7];
        
        x_glo = threadIdx.x / 16;                y_glo = threadIdx.x % 16;

        shmemB[x_glo][y_glo] = B[glo_dex_B];            //load globalB to shmemB
        
        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_glo];
            *((float4*)&tmp_B) = shmemB[__line][y_glo];
            
            hfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo;
    h_store(glo_dex_A)
#endif
}




// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_anyWH_anyL(float4*                   A,
                             float4*                   B,
                             float4*                   dst,
                             const uint                pitch_A,
                             const uint                pitch_B,
                             const uint                Hdst,
                             const uint                HB,
                             const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    half2 sum[8][4];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst && y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B && x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
    }
#endif
}




// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_ABC_anyWH_anyL(float4*                   A,
                                 float4*                   B,
                                 float4*                   C,
                                 float4*                   dst,
                                 const uint                pitch_A,
                                 const uint                pitch_B,
                                 const uint                Hdst,
                                 const uint                HB,
                                 const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    half2 sum[8][4];

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)           h_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)       h_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)       h_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)       h_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)       h_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)       h_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)       h_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)       h_loadC_line(7, glo_dex_A);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst && y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B && x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
    }
#endif
}




// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_anyWH_specL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                Hdst,
                              const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    half2 sum[8][4];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;
    if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
    if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
    if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
    if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
    if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
    if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
    if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
    if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
#endif
}


// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_ABC_anyWH_specL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                Hdst,
                                  const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    half2 sum[8][4];

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)           h_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)       h_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)       h_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)       h_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)       h_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)       h_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)       h_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)       h_loadC_line(7, glo_dex_A);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    if (y_gloA < pitch_B) {
        if (x_gloA < Hdst)          h_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      h_store_one_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)      h_store_one_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)      h_store_one_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)      h_store_one_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)      h_store_one_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)      h_store_one_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)      h_store_one_line(7, glo_dex_A);
    }
#endif
}



// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_specWH_anyL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                HB,
                              const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    half2 sum[8][4];
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        *((float4*)&sum[i][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    size_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    size_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;
    h_store(glo_dex_A)
#endif
}



// last storage (16, 16)
// ���� / �ô� �� is the crucial, reduce memory assess by vectorization
__global__
void decx::gemm::GPUK::cu_GEMM_fp16_ABC_specWH_anyL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                HB,
                                  const uint                __iter)
{
#if __ABOVE_SM_53
    uint x_gloA, y_gloA, x_gloB, y_gloB, x_loc, y_loc;
    size_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[16][128 / 8 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;

    half2 sum[8][4];
    h_loadC(glo_dex_A);

    half tmp_A[8];
    half2 tmp_B[4];
    half2 fma_tmp;

    x_gloA = threadIdx.x / 2 + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 2;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) + blockIdx.y * 16;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint i = 0; i < __iter; ++i)
    {
        *((float4*)&tmp_A) = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A)
            *((float4*)&tmp_A) = A[glo_dex_A];

        x_loc = 8 * (threadIdx.x % 2);            y_loc = (threadIdx.x % 16) / 2;

        *((half*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0];
        *((half*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[1];
        *((half*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[2];
        *((half*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[3];
        *((half*)&(shmemA[x_loc + 4][threadIdx.x / 16]) + y_loc) = tmp_A[4];
        *((half*)&(shmemA[x_loc + 5][threadIdx.x / 16]) + y_loc) = tmp_A[5];
        *((half*)&(shmemA[x_loc + 6][threadIdx.x / 16]) + y_loc) = tmp_A[6];
        *((half*)&(shmemA[x_loc + 7][threadIdx.x / 16]) + y_loc) = tmp_A[7];

        x_loc = threadIdx.x / 16;                y_loc = threadIdx.x % 16;

        if (x_gloB < HB)
            *((float4*)&tmp_A) = B[glo_dex_B];
        shmemB[x_loc][y_loc] = *((float4*)&tmp_A);            //load globalB to shmemB

        __syncthreads();

        glo_dex_A += 2;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 2;
        x_gloB += 16;

#pragma unroll 16
        for (uint __line = 0; __line < 16; ++__line)
        {
            *((float4*)&tmp_A) = shmemA[__line][x_loc];
            *((float4*)&tmp_B) = shmemB[__line][y_loc];

            hfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA;
    h_store(glo_dex_A);
#endif
}