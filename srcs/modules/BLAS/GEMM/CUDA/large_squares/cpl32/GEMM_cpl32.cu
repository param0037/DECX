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


#include "GEMM_cpl32.cuh"


__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_spec(float4*                   A,
                        float4*                   B,
                        float4*                   dst,
                        const uint                pitch_A,       // in float4
                        const uint                pitch_B,       // in float4
                        const uint                __iter)
{
    uint x_glo;
    uint y_glo;
    
    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];
    
    decx::utils::_cuda_vec128 sum[8];
    
    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;
    
#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id) {
        Init_Sum_Union;

        glo_dex_A = ((threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x) * pitch_A + (threadIdx.x % 4) * 2;
        glo_dex_B = (threadIdx.x / 16) * pitch_B + ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
            tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
            tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
            tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1

            x_glo = 4 * (threadIdx.x % 4);            y_glo = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_glo][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_glo + 1][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_glo + 2][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_glo + 3][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[1]._vd.y;
               
            *((double*)&(shmemA[x_glo + 16][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_glo + 17][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_glo + 18][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_glo + 19][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[3]._vd.y;

            x_glo = threadIdx.x / 16;            y_glo = threadIdx.x % 16;

            tmp_B._vf = B[glo_dex_B];
            shmemB[x_glo][y_glo] = tmp_B._vf;            //load globalB to shmemB

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_glo * 2];
                tmp_A[1]._vf = shmemA[__line][x_glo * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_glo * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_glo * 2 + 1];

                tmp_B._vf = shmemB[__line][y_glo];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
        glo_dex_A = x_glo * pitch_B + y_glo * 4 + _lane_id;
        cpl32_store(glo_dex_A);
    }
}



__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_ABC_spec(float4 *                A,
                            float4 *                B,
                            float4 *                C,
                            float4 *                dst,
                            const uint              pitch_A,
                            const uint              pitch_B,
                            const uint              __iter)
{
    uint x_glo;
    uint y_glo;

    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;

#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id) 
    {
        x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
        glo_dex_A = x_glo * pitch_B + y_glo * 4 + _lane_id;
        cpl32_loadC(glo_dex_A);

        glo_dex_A = ((threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x) * pitch_A + (threadIdx.x % 4) * 2;
        glo_dex_B = (threadIdx.x / 16) * pitch_B + ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
            tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
            tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
            tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1

            x_glo = 4 * (threadIdx.x % 4);            y_glo = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_glo][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_glo + 1][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_glo + 2][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_glo + 3][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[1]._vd.y;

            *((double*)&(shmemA[x_glo + 16][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_glo + 17][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_glo + 18][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_glo + 19][(threadIdx.x / 16) * 2]) + y_glo) = tmp_A[3]._vd.y;

            x_glo = threadIdx.x / 16;            y_glo = threadIdx.x % 16;

            tmp_B._vf = B[glo_dex_B];
            shmemB[x_glo][y_glo] = tmp_B._vf;            //load globalB to shmemB

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_glo * 2];
                tmp_A[1]._vf = shmemA[__line][x_glo * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_glo * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_glo * 2 + 1];

                tmp_B._vf = shmemB[__line][y_glo];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_glo = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_glo = (threadIdx.x % 16 + blockIdx.y * 16);
        glo_dex_A = x_glo * pitch_B + y_glo * 4 + _lane_id;
        cpl32_store(glo_dex_A);
    }
}




__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_anyWH_specL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                Hdst,
                              const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;

    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;
    
#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id)
    {
        Init_Sum_Union;

        x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
        y_gloA = (threadIdx.x % 4) * 2;
        x_gloB = (threadIdx.x / 16);
        y_gloB = ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        glo_dex_A = x_gloA * pitch_A + y_gloA;
        glo_dex_B = x_gloB * pitch_B + y_gloB;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = make_float4(0, 0, 0, 0);         tmp_A[1]._vf = make_float4(0, 0, 0, 0);
            tmp_A[2]._vf = make_float4(0, 0, 0, 0);         tmp_A[3]._vf = make_float4(0, 0, 0, 0);
            tmp_B._vf = make_float4(0, 0, 0, 0);

            if (x_gloA < Hdst) {
                tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
                tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
            }
            if (x_gloA + 4 < Hdst) {
                tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
                tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1
            }

            x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_loc][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_loc + 1][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_loc + 2][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_loc + 3][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.y;

            *((double*)&(shmemA[x_loc + 16][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_loc + 17][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_loc + 18][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_loc + 19][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.y;

            x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

            if (y_gloB < pitch_B)   tmp_B._vf = B[glo_dex_B];
            shmemB[x_loc][y_loc] = tmp_B._vf;            //load globalB to shmemB

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;
            y_gloA += 8;
            x_gloB += 16;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_loc * 2];
                tmp_A[1]._vf = shmemA[__line][x_loc * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_loc * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_loc * 2 + 1];

                tmp_B._vf = shmemB[__line][y_loc];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;

        if (y_gloA < pitch_B) {
            if (x_gloA < Hdst)          cpl32_store_one_line(0, glo_dex_A);
            if (x_gloA + 1 < Hdst)      cpl32_store_one_line(1, glo_dex_A);
            if (x_gloA + 2 < Hdst)      cpl32_store_one_line(2, glo_dex_A);
            if (x_gloA + 3 < Hdst)      cpl32_store_one_line(3, glo_dex_A);
            if (x_gloA + 4 < Hdst)      cpl32_store_one_line(4, glo_dex_A);
            if (x_gloA + 5 < Hdst)      cpl32_store_one_line(5, glo_dex_A);
            if (x_gloA + 6 < Hdst)      cpl32_store_one_line(6, glo_dex_A);
            if (x_gloA + 7 < Hdst)      cpl32_store_one_line(7, glo_dex_A);
        }
    }
}




__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_ABC_anyWH_specL(float4*                   A,
                                   float4*                   B,
                                   float4*                   C,
                                   float4*                   dst,
                                   const uint                pitch_A,
                                   const uint                pitch_B,
                                   const uint                Hdst,
                                   const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;

    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;

#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id)
    {
        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;

        if (y_gloA < pitch_B) {
            if (x_gloA < Hdst)          { sum[0]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 1 < Hdst)      { sum[1]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 2 < Hdst)      { sum[2]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 3 < Hdst)      { sum[3]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 4 < Hdst)      { sum[4]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 5 < Hdst)      { sum[5]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 6 < Hdst)      { sum[6]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 7 < Hdst)      { sum[7]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
        }

        x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
        y_gloA = (threadIdx.x % 4) * 2;
        x_gloB = (threadIdx.x / 16);
        y_gloB = ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        glo_dex_A = x_gloA * pitch_A + y_gloA;
        glo_dex_B = x_gloB * pitch_B + y_gloB;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = make_float4(0, 0, 0, 0);         tmp_A[1]._vf = make_float4(0, 0, 0, 0);
            tmp_A[2]._vf = make_float4(0, 0, 0, 0);         tmp_A[3]._vf = make_float4(0, 0, 0, 0);
            tmp_B._vf = make_float4(0, 0, 0, 0);

            if (x_gloA < Hdst) {
                tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
                tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
            }
            if (x_gloA + 4 < Hdst) {
                tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
                tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1
            }

            x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_loc][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_loc + 1][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_loc + 2][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_loc + 3][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.y;

            *((double*)&(shmemA[x_loc + 16][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_loc + 17][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_loc + 18][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_loc + 19][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.y;

            x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

            if (y_gloB < pitch_B)   tmp_B._vf = B[glo_dex_B];
            shmemB[x_loc][y_loc] = tmp_B._vf;            //load globalB to shmemB

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;
            y_gloA += 8;
            x_gloB += 16;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_loc * 2];
                tmp_A[1]._vf = shmemA[__line][x_loc * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_loc * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_loc * 2 + 1];

                tmp_B._vf = shmemB[__line][y_loc];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;

        if (y_gloA < pitch_B) {
            if (x_gloA < Hdst)          cpl32_store_one_line(0, glo_dex_A);
            if (x_gloA + 1 < Hdst)      cpl32_store_one_line(1, glo_dex_A);
            if (x_gloA + 2 < Hdst)      cpl32_store_one_line(2, glo_dex_A);
            if (x_gloA + 3 < Hdst)      cpl32_store_one_line(3, glo_dex_A);
            if (x_gloA + 4 < Hdst)      cpl32_store_one_line(4, glo_dex_A);
            if (x_gloA + 5 < Hdst)      cpl32_store_one_line(5, glo_dex_A);
            if (x_gloA + 6 < Hdst)      cpl32_store_one_line(6, glo_dex_A);
            if (x_gloA + 7 < Hdst)      cpl32_store_one_line(7, glo_dex_A);
        }
    }
}



__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_anyWH_anyL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,        // in float4
                              const uint                pitch_B,        // in float4
                              const uint                Hdst,
                              const uint                HB,
                              const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;
    
    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;
    
#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id) 
    {
        Init_Sum_Union;

        x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
        y_gloA = (threadIdx.x % 4) * 2;
        x_gloB = (threadIdx.x / 16);
        y_gloB = ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        glo_dex_A = x_gloA * pitch_A + y_gloA;
        glo_dex_B = x_gloB * pitch_B + y_gloB;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = make_float4(0, 0, 0, 0);         tmp_A[1]._vf = make_float4(0, 0, 0, 0);
            tmp_A[2]._vf = make_float4(0, 0, 0, 0);         tmp_A[3]._vf = make_float4(0, 0, 0, 0);
            tmp_B._vf = make_float4(0, 0, 0, 0);

            if (y_gloA < pitch_A) {
                if (x_gloA < Hdst) {
                    tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
                    tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
                }
                if (x_gloA + 4 < Hdst) {
                    tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
                    tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1
                }
            }

            x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_loc][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_loc + 1][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_loc + 2][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_loc + 3][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.y;
                                                                        
            *((double*)&(shmemA[x_loc + 16][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_loc + 17][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_loc + 18][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_loc + 19][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.y;

            x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

            if (x_gloB < HB) {
                if (y_gloB < pitch_B)   tmp_B._vf = B[glo_dex_B];
            }

            shmemB[x_loc][y_loc] = tmp_B._vf;

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;
            y_gloA += 8;
            x_gloB += 16;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_loc * 2];
                tmp_A[1]._vf = shmemA[__line][x_loc * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_loc * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_loc * 2 + 1];

                tmp_B._vf = shmemB[__line][y_loc];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;

        if (y_gloA < pitch_B) {
            if (x_gloA < Hdst)          cpl32_store_one_line(0, glo_dex_A);
            if (x_gloA + 1 < Hdst)      cpl32_store_one_line(1, glo_dex_A);
            if (x_gloA + 2 < Hdst)      cpl32_store_one_line(2, glo_dex_A);
            if (x_gloA + 3 < Hdst)      cpl32_store_one_line(3, glo_dex_A);
            if (x_gloA + 4 < Hdst)      cpl32_store_one_line(4, glo_dex_A);
            if (x_gloA + 5 < Hdst)      cpl32_store_one_line(5, glo_dex_A);
            if (x_gloA + 6 < Hdst)      cpl32_store_one_line(6, glo_dex_A);
            if (x_gloA + 7 < Hdst)      cpl32_store_one_line(7, glo_dex_A);
        }
    }
}



__global__ void 
decx::gemm::GPUK::cu_GEMM_cpl32_ABC_anyWH_anyL(float4* __restrict  A,
                                               float4* __restrict  B,
                                               float4* __restrict  C,
                                               float4*                   dst,
                                               const uint                pitch_A,
                                               const uint                pitch_B,
                                               const uint                Hdst,
                                               const uint                HB,
                                               const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;

    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;

#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id)
    {
        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;

        if (y_gloA < pitch_B) {
            if (x_gloA < Hdst)      { sum[0]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 1 < Hdst)  { sum[1]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 2 < Hdst)  { sum[2]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 3 < Hdst)  { sum[3]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 4 < Hdst)  { sum[4]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 5 < Hdst)  { sum[5]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 6 < Hdst)  { sum[6]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
            if (x_gloA + 7 < Hdst)  { sum[7]._vf = C[glo_dex_A];       glo_dex_A += pitch_B; }
        }

        x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
        y_gloA = (threadIdx.x % 4) * 2;
        x_gloB = (threadIdx.x / 16);
        y_gloB = ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        glo_dex_A = x_gloA * pitch_A + y_gloA;
        glo_dex_B = x_gloB * pitch_B + y_gloB;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = make_float4(0, 0, 0, 0);         tmp_A[1]._vf = make_float4(0, 0, 0, 0);
            tmp_A[2]._vf = make_float4(0, 0, 0, 0);         tmp_A[3]._vf = make_float4(0, 0, 0, 0);
            tmp_B._vf = make_float4(0, 0, 0, 0);

            if (y_gloA < pitch_A) {
                if (x_gloA < Hdst) {
                    tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
                    tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
                }
                if (x_gloA + 4 < Hdst) {
                    tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
                    tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1
                }
            }

            x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_loc][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_loc + 1][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_loc + 2][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_loc + 3][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.y;

            *((double*)&(shmemA[x_loc + 16][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_loc + 17][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_loc + 18][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_loc + 19][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.y;

            x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

            if (x_gloB < HB) {
                if (y_gloB < pitch_B)   tmp_B._vf = B[glo_dex_B];
            }

            shmemB[x_loc][y_loc] = tmp_B._vf;

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;
            y_gloA += 8;
            x_gloB += 16;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_loc * 2];
                tmp_A[1]._vf = shmemA[__line][x_loc * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_loc * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_loc * 2 + 1];

                tmp_B._vf = shmemB[__line][y_loc];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;

        if (y_gloA < pitch_B) {
            if (x_gloA < Hdst)          cpl32_store_one_line(0, glo_dex_A);
            if (x_gloA + 1 < Hdst)      cpl32_store_one_line(1, glo_dex_A);
            if (x_gloA + 2 < Hdst)      cpl32_store_one_line(2, glo_dex_A);
            if (x_gloA + 3 < Hdst)      cpl32_store_one_line(3, glo_dex_A);
            if (x_gloA + 4 < Hdst)      cpl32_store_one_line(4, glo_dex_A);
            if (x_gloA + 5 < Hdst)      cpl32_store_one_line(5, glo_dex_A);
            if (x_gloA + 6 < Hdst)      cpl32_store_one_line(6, glo_dex_A);
            if (x_gloA + 7 < Hdst)      cpl32_store_one_line(7, glo_dex_A);
        }
    }
}




__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_specWH_anyL(float4*                   A,
                              float4*                   B,
                              float4*                   dst,
                              const uint                pitch_A,
                              const uint                pitch_B,
                              const uint                HB,
                              const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;

    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;

#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id)
    {
        Init_Sum_Union;

        x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
        y_gloA = (threadIdx.x % 4) * 2;
        x_gloB = (threadIdx.x / 16);
        y_gloB = ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        glo_dex_A = x_gloA * pitch_A + y_gloA;
        glo_dex_B = x_gloB * pitch_B + y_gloB;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = make_float4(0, 0, 0, 0);         tmp_A[1]._vf = make_float4(0, 0, 0, 0);
            tmp_A[2]._vf = make_float4(0, 0, 0, 0);         tmp_A[3]._vf = make_float4(0, 0, 0, 0);
            tmp_B._vf = make_float4(0, 0, 0, 0);

            tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
            tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
            tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
            tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1

            x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_loc][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_loc + 1][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_loc + 2][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_loc + 3][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.y;

            *((double*)&(shmemA[x_loc + 16][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_loc + 17][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_loc + 18][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_loc + 19][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.y;

            x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

            tmp_B._vf = B[glo_dex_B];
            shmemB[x_loc][y_loc] = tmp_B._vf;            //load globalB to shmemB

            shmemB[x_loc][y_loc] = tmp_B._vf;

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;
            y_gloA += 8;
            x_gloB += 16;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_loc * 2];
                tmp_A[1]._vf = shmemA[__line][x_loc * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_loc * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_loc * 2 + 1];

                tmp_B._vf = shmemB[__line][y_loc];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;
        cpl32_store(glo_dex_A);
    }
}



__global__
void decx::gemm::GPUK::cu_GEMM_cpl32_ABC_specWH_anyL(float4*                   A,
                                  float4*                   B,
                                  float4*                   C,
                                  float4*                   dst,
                                  const uint                pitch_A,
                                  const uint                pitch_B,
                                  const uint                HB,
                                  const uint                __iter)
{
    uint x_gloA, y_gloA, x_gloB, y_gloB;
    uint x_loc, y_loc;

    __shared__ float4 shmemA[32][128 / 4 + 1];
    __shared__ float4 shmemB[16][128 / 8 + 1];

    decx::utils::_cuda_vec128 sum[8];

    decx::utils::_cuda_vec128 tmp_A[4];
    decx::utils::_cuda_vec128 tmp_B;

    size_t glo_dex_A = 0, glo_dex_B = 0;

#pragma unroll 4
    for (int _lane_id = 0; _lane_id < 4; ++_lane_id)
    {
        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;
        cpl32_loadC(glo_dex_A);

        x_gloA = (threadIdx.x / 16) * 8 + ((threadIdx.x / 4) % 4) + 128 * blockIdx.x;
        y_gloA = (threadIdx.x % 4) * 2;
        x_gloB = (threadIdx.x / 16);
        y_gloB = ((threadIdx.x % 16) + blockIdx.y * 16) * 4 + _lane_id;

        glo_dex_A = x_gloA * pitch_A + y_gloA;
        glo_dex_B = x_gloB * pitch_B + y_gloB;

        for (uint i = 0; i < __iter; ++i)
        {
            tmp_A[0]._vf = make_float4(0, 0, 0, 0);         tmp_A[1]._vf = make_float4(0, 0, 0, 0);
            tmp_A[2]._vf = make_float4(0, 0, 0, 0);         tmp_A[3]._vf = make_float4(0, 0, 0, 0);
            tmp_B._vf = make_float4(0, 0, 0, 0);

            tmp_A[0]._vf = A[glo_dex_A];                        // lane 0
            tmp_A[1]._vf = A[glo_dex_A + 1];                    // lane 1
            tmp_A[2]._vf = A[glo_dex_A + pitch_A * 4];          // lane 0
            tmp_A[3]._vf = A[glo_dex_A + pitch_A * 4 + 1];      // lane 1

            x_loc = 4 * (threadIdx.x % 4);            y_loc = (threadIdx.x % 16) / 4;

            *((double*)&(shmemA[x_loc][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.x;
            *((double*)&(shmemA[x_loc + 1][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[0]._vd.y;
            *((double*)&(shmemA[x_loc + 2][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.x;
            *((double*)&(shmemA[x_loc + 3][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[1]._vd.y;

            *((double*)&(shmemA[x_loc + 16][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.x;
            *((double*)&(shmemA[x_loc + 17][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[2]._vd.y;
            *((double*)&(shmemA[x_loc + 18][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.x;
            *((double*)&(shmemA[x_loc + 19][(threadIdx.x / 16) * 2]) + y_loc) = tmp_A[3]._vd.y;

            x_loc = threadIdx.x / 16;            y_loc = threadIdx.x % 16;

            tmp_B._vf = B[glo_dex_B];
            shmemB[x_loc][y_loc] = tmp_B._vf;            //load globalB to shmemB

            shmemB[x_loc][y_loc] = tmp_B._vf;

            __syncthreads();

            glo_dex_A += 8;
            glo_dex_B += 16 * pitch_B;
            y_gloA += 8;
            x_gloB += 16;

#pragma unroll 16
            for (uint __line = 0; __line < 16; ++__line)
            {
                tmp_A[0]._vf = shmemA[__line][x_loc * 2];
                tmp_A[1]._vf = shmemA[__line][x_loc * 2 + 1];
                tmp_A[2]._vf = shmemA[__line + 16][x_loc * 2];
                tmp_A[3]._vf = shmemA[__line + 16][x_loc * 2 + 1];

                tmp_B._vf = shmemB[__line][y_loc];

                cpl32fma_8x8;
            }
            __syncthreads();
        }

        x_gloA = (threadIdx.x / 16 + blockIdx.x * 16) * 8;
        y_gloA = (threadIdx.x % 16 + blockIdx.y * 16) * 4 + _lane_id;
        glo_dex_A = x_gloA * pitch_B + y_gloA;
        cpl32_store(glo_dex_A);
    }
}