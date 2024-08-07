/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "GEMM_fp32.cuh"



__global__
void decx::gemm::GPUK::cu_GEMM_fp32_spec(const float4* __restrict  A,
                                         const float4* __restrict  B,
                                         float4* __restrict        dst,
                                         const uint32_t            pitch_A,
                                         const uint32_t            pitch_B,
                                         const uint32_t            __iter)
{
    uint32_t x_glo;
    uint32_t y_glo;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum;

    float4 tmp_A[2];
    float4 tmp_B[2];
    
    uint64_t glo_dex_A = ((threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x) * pitch_A + threadIdx.x % 4;
    uint64_t glo_dex_B = ((threadIdx.x % 16) * 2 + blockIdx.y * 32) + pitch_B * (threadIdx.x / 16);
    
    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = A[glo_dex_A];
        tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        
        x_glo = 4 * (threadIdx.x % 4);            y_glo = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0].x;
        *((float*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[0].y;
        *((float*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[0].z;
        *((float*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[0].w;

        *((float*)&(shmemA[x_glo + 16][threadIdx.x / 16]) + y_glo) = tmp_A[1].x;
        *((float*)&(shmemA[x_glo + 17][threadIdx.x / 16]) + y_glo) = tmp_A[1].y;
        *((float*)&(shmemA[x_glo + 18][threadIdx.x / 16]) + y_glo) = tmp_A[1].z;
        *((float*)&(shmemA[x_glo + 19][threadIdx.x / 16]) + y_glo) = tmp_A[1].w;

        x_glo = threadIdx.x / 16;            y_glo = threadIdx.x % 16;

        tmp_B[0] = B[glo_dex_B];
        tmp_B[1] = B[glo_dex_B + 1];
        shmemB[x_glo][y_glo] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_glo + 16][y_glo] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_glo];
            tmp_A[1] = shmemA[__line + 16][x_glo];

            tmp_B[0] = shmemB[__line][y_glo];
            tmp_B[1] = shmemB[__line + 16][y_glo];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo * 2;
    s_store(glo_dex_A);
}





__global__
void decx::gemm::GPUK::cu_GEMM_fp32_ABC_spec(const float4* __restrict   A,
                                             const float4* __restrict   B,
                                             const float4* __restrict   C,
                                             float4 *                   dst,
                                             const uint32_t             pitch_A,
                                             const uint32_t             pitch_B,
                                             const uint32_t             __iter)
{
    uint32_t x_glo, y_glo;
    uint64_t glo_dex_A;

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo * 2;

    float4 sum[8][2];

    s_loadC(glo_dex_A);            // initialize sum with C

    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 tmp_A[2], tmp_B[2];
    
    glo_dex_A = ((threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x) * pitch_A + threadIdx.x % 4;
    uint64_t glo_dex_B = ((threadIdx.x % 16) * 2 + blockIdx.y * 32) + pitch_B * (threadIdx.x / 16);
    
    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = A[glo_dex_A];
        tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        
        x_glo = 4 * (threadIdx.x % 4);            y_glo = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_glo][threadIdx.x / 16]) + y_glo) = tmp_A[0].x;
        *((float*)&(shmemA[x_glo + 1][threadIdx.x / 16]) + y_glo) = tmp_A[0].y;
        *((float*)&(shmemA[x_glo + 2][threadIdx.x / 16]) + y_glo) = tmp_A[0].z;
        *((float*)&(shmemA[x_glo + 3][threadIdx.x / 16]) + y_glo) = tmp_A[0].w;

        *((float*)&(shmemA[x_glo + 16][threadIdx.x / 16]) + y_glo) = tmp_A[1].x;
        *((float*)&(shmemA[x_glo + 17][threadIdx.x / 16]) + y_glo) = tmp_A[1].y;
        *((float*)&(shmemA[x_glo + 18][threadIdx.x / 16]) + y_glo) = tmp_A[1].z;
        *((float*)&(shmemA[x_glo + 19][threadIdx.x / 16]) + y_glo) = tmp_A[1].w;

        x_glo = threadIdx.x / 16;            y_glo = threadIdx.x % 16;

        tmp_B[0] = B[glo_dex_B];
        tmp_B[1] = B[glo_dex_B + 1];
        shmemB[x_glo][y_glo] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_glo + 16][y_glo] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_glo];
            tmp_A[1] = shmemA[__line + 16][x_glo];

            tmp_B[0] = shmemB[__line][y_glo];
            tmp_B[1] = shmemB[__line + 16][y_glo];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_glo * pitch_B + y_glo * 2;
    s_store(glo_dex_A);
}




__global__
void decx::gemm::GPUK::cu_GEMM_fp32_anyWH_specL(const float4* __restrict    A,
                                                const float4* __restrict    B,
                                                float4*                     dst,
                                                const uint32_t              pitch_A,
                                                const uint32_t              pitch_B,
                                                const uint32_t              Hdst,
                                                const uint32_t              __iter)
{
    uint32_t x_gloA, y_gloA, x_gloB, y_gloB;
    uint32_t x_loc, y_loc;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum;

    float4 tmp_A[2] = { make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0) };
    float4 tmp_B[2] = { make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0) };
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    uint64_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    uint64_t glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint32_t i = 0; i < __iter; ++i)
    {
        if (x_gloA < Hdst)
            tmp_A[0] = A[glo_dex_A];
        if (x_gloA + 4 < Hdst)
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            tmp_B[0] = B[glo_dex_B];
        if (y_gloB + 1 < pitch_B)
            tmp_B[1] = B[glo_dex_B + 1];
        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        
#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}



__global__
void decx::gemm::GPUK::cu_GEMM_fp32_ABC_anyWH_specL(const float4* __restrict      A,
                                                    const float4* __restrict      B,
                                                    const float4* __restrict      C,
                                                    float4* __restrict            dst,
                                                    const uint32_t                pitch_A,
                                                    const uint32_t                pitch_B,
                                                    const uint32_t                Hdst,
                                                    const uint32_t                __iter)
{
    uint32_t x_gloA, y_gloA, x_gloB, y_gloB;
    uint32_t x_loc, y_loc;
    uint64_t glo_dex_A, glo_dex_B;

    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    float4 sum[8][2];
    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)              s_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)          s_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)          s_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)          s_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)          s_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)          s_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)          s_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)          s_loadC_line(7, glo_dex_A);
    }

    float4 tmp_A[2], tmp_B[2];

    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (x_gloA < Hdst)
            tmp_A[0] = A[glo_dex_A];
        if (x_gloA + 4 < Hdst)
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];

        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (y_gloB < pitch_B)
            tmp_B[0] = B[glo_dex_B];
        if (y_gloB + 1 < pitch_B)
            tmp_B[1] = B[glo_dex_B + 1];

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];

        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;

#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}




__global__
void decx::gemm::GPUK::cu_GEMM_fp32_anyWH_anyL(const float4* __restrict   A,
                                               const float4* __restrict   B,
                                               float4* __restrict         dst,
                                               const uint32_t             pitch_A,
                                               const uint32_t             pitch_B,
                                               const uint32_t             Hdst,
                                               const uint32_t             HB,
                                               const uint32_t             __iter)
{
    uint32_t x_gloA, y_gloA, x_gloB, y_gloB;
    uint32_t x_loc, y_loc;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    float4 sum[8][2];
    Init_Sum;

    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    uint64_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    uint64_t glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            if (x_gloA < Hdst)
                tmp_A[0] = A[glo_dex_A];
            if (x_gloA + 4 < Hdst)
                tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            if (y_gloB < pitch_B)
                tmp_B[0] = B[glo_dex_B];
            if (y_gloB + 1 < pitch_B)
                tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}



__global__
void decx::gemm::GPUK::cu_GEMM_fp32_ABC_anyWH_anyL(const float4* __restrict  A,
                                                   const float4* __restrict  B,
                                                   const float4* __restrict  C,
                                                   float4* __restrict        dst,
                                                   const uint32_t            pitch_A,
                                                   const uint32_t            pitch_B,
                                                   const uint32_t            Hdst,
                                                   const uint32_t            HB,
                                                   const uint32_t            __iter)
{
    uint32_t x_gloA, y_gloA, x_gloB, y_gloB;
    uint32_t x_loc, y_loc;
    uint64_t glo_dex_A, glo_dex_B;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    
    float4 sum[8][2];
    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)              s_loadC_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)          s_loadC_line(1, glo_dex_A);
        if (x_gloA + 2 < Hdst)          s_loadC_line(2, glo_dex_A);
        if (x_gloA + 3 < Hdst)          s_loadC_line(3, glo_dex_A);
        if (x_gloA + 4 < Hdst)          s_loadC_line(4, glo_dex_A);
        if (x_gloA + 5 < Hdst)          s_loadC_line(5, glo_dex_A);
        if (x_gloA + 6 < Hdst)          s_loadC_line(6, glo_dex_A);
        if (x_gloA + 7 < Hdst)          s_loadC_line(7, glo_dex_A);
    }
    
    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            if (x_gloA < Hdst)
                tmp_A[0] = A[glo_dex_A];
            if (x_gloA + 4 < Hdst)
                tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            if (y_gloB < pitch_B)
                tmp_B[0] = B[glo_dex_B];
            if (y_gloB + 1 < pitch_B)
                tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;

    if (y_gloA * 2 < pitch_B) {
        if (x_gloA < Hdst)          s_store_one_line(0, glo_dex_A);
        if (x_gloA + 1 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 2 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 3 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 4 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 5 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 6 < Hdst)      s_store_one_line(0, glo_dex_A);
        if (x_gloA + 7 < Hdst)      s_store_one_line(0, glo_dex_A);
    }
}



__global__
void decx::gemm::GPUK::cu_GEMM_fp32_specWH_anyL(const float4* __restrict      A,
                                                const float4* __restrict      B,
                                                float4* __restrict            dst,
                                                const uint32_t                pitch_A,
                                                const uint32_t                pitch_B,
                                                const uint32_t                HB,
                                                const uint32_t                __iter)
{
    uint32_t x_gloA, y_gloA, x_gloB, y_gloB;
    uint32_t x_loc, y_loc;

    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];

    float4 sum[8][2];
    Init_Sum;

        float4 tmp_A[2], tmp_B[2];

    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    uint64_t glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    uint64_t glo_dex_B = y_gloB + pitch_B * x_gloB;

    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            tmp_A[0] = A[glo_dex_A];
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }

        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            tmp_B[0] = B[glo_dex_B];
            tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];

        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;

#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    s_store(glo_dex_A);
}



__global__
void decx::gemm::GPUK::cu_GEMM_fp32_ABC_specWH_anyL(const float4* __restrict    A,
                                                    const float4* __restrict    B,
                                                    const float4* __restrict    C,
                                                    float4* __restrict          dst,
                                                    const uint32_t              pitch_A,
                                                    const uint32_t              pitch_B,
                                                    const uint32_t              HB,
                                                    const uint32_t              __iter)
{
    uint32_t x_gloA, y_gloA, x_gloB, y_gloB;
    uint32_t x_loc, y_loc;
    uint64_t glo_dex_A, glo_dex_B;
    
    __shared__ float4 shmemA[32][128 / 8 + 1];
    __shared__ float4 shmemB[32][128 / 8 + 1];
    
    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    
    float4 sum[8][2];
    s_loadC(glo_dex_A);
    
    float4 tmp_A[2], tmp_B[2];
    
    x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
    y_gloA = threadIdx.x % 4;
    glo_dex_A = x_gloA * pitch_A + y_gloA;

    x_gloB = threadIdx.x / 16;
    y_gloB = (threadIdx.x % 16) * 2 + blockIdx.y * 32;
    glo_dex_B = y_gloB + pitch_B * x_gloB;
    
    for (uint32_t i = 0; i < __iter; ++i)
    {
        tmp_A[0] = make_float4(0, 0, 0, 0);         tmp_A[1] = make_float4(0, 0, 0, 0);
        tmp_B[0] = make_float4(0, 0, 0, 0);         tmp_B[1] = make_float4(0, 0, 0, 0);

        if (y_gloA < pitch_A) {
            tmp_A[0] = A[glo_dex_A];
            tmp_A[1] = A[glo_dex_A + pitch_A * 4];
        }
        
        x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

        *((float*)&(shmemA[x_loc][threadIdx.x / 16]) + y_loc) = tmp_A[0].x;
        *((float*)&(shmemA[x_loc + 1][threadIdx.x / 16]) + y_loc) = tmp_A[0].y;
        *((float*)&(shmemA[x_loc + 2][threadIdx.x / 16]) + y_loc) = tmp_A[0].z;
        *((float*)&(shmemA[x_loc + 3][threadIdx.x / 16]) + y_loc) = tmp_A[0].w;

        *((float*)&(shmemA[x_loc + 16][threadIdx.x / 16]) + y_loc) = tmp_A[1].x;
        *((float*)&(shmemA[x_loc + 17][threadIdx.x / 16]) + y_loc) = tmp_A[1].y;
        *((float*)&(shmemA[x_loc + 18][threadIdx.x / 16]) + y_loc) = tmp_A[1].z;
        *((float*)&(shmemA[x_loc + 19][threadIdx.x / 16]) + y_loc) = tmp_A[1].w;

        x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

        if (x_gloB < HB) {
            tmp_B[0] = B[glo_dex_B];
            tmp_B[1] = B[glo_dex_B + 1];
        }

        shmemB[x_loc][y_loc] = tmp_B[0];            //load globalB to shmemB
        shmemB[x_loc + 16][y_loc] = tmp_B[1];
        
        __syncthreads();

        glo_dex_A += 4;
        glo_dex_B += 16 * pitch_B;
        y_gloA += 4;
        x_gloB += 16;
        
#pragma unroll 16
        for (uint32_t __line = 0; __line < 16; ++__line)
        {
            tmp_A[0] = shmemA[__line][x_loc];
            tmp_A[1] = shmemA[__line + 16][x_loc];

            tmp_B[0] = shmemB[__line][y_loc];
            tmp_B[1] = shmemB[__line + 16][y_loc];

            sfma_8x8;
        }
        __syncthreads();
    }

    x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
    y_gloA = (threadIdx.x % 16 + blockIdx.y * 16);
    glo_dex_A = x_gloA * pitch_B + y_gloA * 2;
    s_store(glo_dex_A);
}
