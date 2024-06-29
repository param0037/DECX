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


#ifndef _CUDA_NLM_BGR_KEEP_ALPHA_CUH_
#define _CUDA_NLM_BGR_KEEP_ALPHA_CUH_

#include "../../core/basic.h"
#include "NLM_device_functions.cuh"


__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \96 floats     16 uchar4s
* -----------------------------------
* |                                 |        16 uchar4s
* |         -----------------       |
* |         |               |       |
* |    apron|     constant  |       |        48 floats  => __shared__ uchar4 src_frag[32][80]
* |         |               |       |
* |         -----------------       |
* |                                 |
* -----------------------------------
*/
void cu_NLM_r16_BGR_KPAL_N3x3(float4*               src,
                              float4*               dst,
                              const uint            pitch_src,
                              const uint            pitch_dst,
                              const uint            eq_total_ker_len,
                              const uint            eq_Wker,
                              const uint2           eq_kernel_shift,
                              const float           h_2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ uchar4 src_frag[48][96 + 1];

    /**
    *             0   1   2   3   4   5   6   7   8   9  10  11
    *    time  [  x   x   x   x   x   x   x   x   x   x   x   x  ]
    *     1     |load from src  |    
    *     2     |  temp for a vec4 weights                       |
    *     3     |  denoised_res                                  |
    */
    float reg[12], 
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    float3 calc_tmp, normalize_coef[4] = {
        make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0) };

    int dx, dy, load_N;
    uchar4 center[3][6], neigbour_window[3][6];
    float4 res;

    uint glo_dex = idx * pitch_src + idy;           
    *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar4(0);

    glo_dex += 16 * pitch_src;                      
    *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar4(16);

    glo_dex += 16 * pitch_src;                      
    *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar4(32);

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;       
        *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar4(0);

        glo_dex += 16 * pitch_src;                  
        *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar4(16);

        glo_dex += 16 * pitch_src;                  
        *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar4(32);
    }

    __syncthreads();

    // load values for center
    for (dx = 0; dx < 6; ++dx) {
        center[0][dx] = src_frag[threadIdx.x + 15][4 * threadIdx.y + 15 + dx];
        center[1][dx] = src_frag[threadIdx.x + 16][4 * threadIdx.y + 15 + dx];
        center[2][dx] = src_frag[threadIdx.x + 17][4 * threadIdx.y + 15 + dx];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 6; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][4 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][4 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][4 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_3x6_uchar4((float*)(&neigbour_window[0]));
            reg_shift_3x6_uchar4((float*)(&neigbour_window[1]));
            reg_shift_3x6_uchar4((float*)(&neigbour_window[2]));

            neigbour_window[0][5] = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 5];
            neigbour_window[1][5] = src_frag[threadIdx.x + dx + 1][4 * (threadIdx.y) + dy + 5];
            neigbour_window[2][5] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 5];
        }

        // element one
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][0], neigbour_window[load_N][0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[0], &normalize_coef[0], tot_div, &calc_tmp, &neigbour_window[1][1]);
        
        // element two
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[3], &normalize_coef[1], tot_div, &calc_tmp, &neigbour_window[1][2]);

        // element three
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[6], &normalize_coef[2], tot_div, &calc_tmp, &neigbour_window[1][3]);

        // element four
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[9], &normalize_coef[3], tot_div, &calc_tmp, &neigbour_window[1][4]);
    }

    cu_NLM_normalization_BGR_kpal(reg, normalize_coef, (uchar4*)&res, &center[1][1]);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}


__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \96 floats     16 uchar4s
* -----------------------------------
* |                                 |        16 uchar4s
* |         -----------------       |
* |         |               |       |
* |    apron|     constant  |       |        48 floats  => __shared__ uchar4 src_frag[32][80]
* |         |               |       |
* |         -----------------       |
* |                                 |
* -----------------------------------
*/
void cu_NLM_r16_BGR_KPAL_N5x5(float4*               src,
                              float4*               dst,
                              const uint            pitch_src,
                              const uint            pitch_dst,
                              const uint            eq_total_ker_len,
                              const uint            eq_Wker,
                              const uint2           eq_kernel_shift,
                              const float           h_2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ uchar4 src_frag[48][96 + 1];

    /**
    *             0   1   2   3   4   5   6   7   8   9  10  11
    *    time  [  x   x   x   x   x   x   x   x   x   x   x   x  ]
    *     1     |load from src  |    
    *     2     |  temp for a vec4 weights                       |
    *     3     |  denoised_res                                  |
    */
    float reg[12], 
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    float3 calc_tmp, normalize_coef[4] = {
        make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0) };

    int dx, dy, load_N;
    uchar4 center[5][8], neigbour_window[5][8];
    float4 res;

    uint glo_dex = idx * pitch_src + idy;           *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar4(0);

    glo_dex += 16 * pitch_src;                      *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar4(16);

    glo_dex += 16 * pitch_src;                      *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar4(32);

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;       *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar4(0);

        glo_dex += 16 * pitch_src;                  *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar4(16);

        glo_dex += 16 * pitch_src;                  *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar4(32);
    }

    __syncthreads();

    // load values for center
    for (dx = 0; dx < 8; ++dx) {
        center[0][dx] = src_frag[threadIdx.x + 14][4 * threadIdx.y + 14 + dx];
        center[1][dx] = src_frag[threadIdx.x + 15][4 * threadIdx.y + 14 + dx];
        center[2][dx] = src_frag[threadIdx.x + 16][4 * threadIdx.y + 14 + dx];
        center[3][dx] = src_frag[threadIdx.x + 17][4 * threadIdx.y + 14 + dx];
        center[4][dx] = src_frag[threadIdx.x + 18][4 * threadIdx.y + 14 + dx];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 8; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][4 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][4 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][4 * threadIdx.y + dy + load_N];
                neigbour_window[3][load_N] = src_frag[threadIdx.x + dx + 3][4 * threadIdx.y + dy + load_N];
                neigbour_window[4][load_N] = src_frag[threadIdx.x + dx + 4][4 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_5x8_uchar4((float*)(&neigbour_window[0]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[1]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[2]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[3]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[4]));

            neigbour_window[0][7] = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 7];
            neigbour_window[1][7] = src_frag[threadIdx.x + dx + 1][4 * (threadIdx.y) + dy + 7];
            neigbour_window[2][7] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 7];
            neigbour_window[3][7] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 7];
            neigbour_window[4][7] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 7];
        }

        // element one
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][0], neigbour_window[load_N][0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[0], &normalize_coef[0], tot_div, &calc_tmp, &neigbour_window[2][2]);
        
        // element two
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[3], &normalize_coef[1], tot_div, &calc_tmp, &neigbour_window[2][3]);

        // element three
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][6], neigbour_window[load_N][6], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[6], &normalize_coef[2], tot_div, &calc_tmp, &neigbour_window[2][4]);

        // element four
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][6], neigbour_window[load_N][6], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][7], neigbour_window[load_N][7], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[9], &normalize_coef[3], tot_div, &calc_tmp, &neigbour_window[2][5]);
    }

    cu_NLM_normalization_BGR_kpal(reg, normalize_coef, (uchar4*)&res, &center[2][2]);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \80 floats     8 uchar4s
* -----------------------------------
* |                                 |        8 uchar4s
* |         -----------------       |
* |         |               |       |
* |    apron|     constant  |       |        32 floats  => __shared__ uchar4 src_frag[32][80]
* |         |               |       |
* |         -----------------       |
* |                                 |
* -----------------------------------
*/
void cu_NLM_r8_BGR_KPAL_N3x3(float4*               src,
                             float4*               dst,
                             const uint            pitch_src,
                             const uint            pitch_dst,
                             const uint            eq_total_ker_len,
                             const uint            eq_Wker,
                             const uint2           eq_kernel_shift,
                             const float           h_2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ uchar4 src_frag[32][80 + 1];

    /**
    *             0   1   2   3   4   5   6   7   8   9  10  11
    *    time  [  x   x   x   x   x   x   x   x   x   x   x   x  ]
    *     1     |load from src  |    
    *     2     |  temp for a vec4 weights                       |
    *     3     |  denoised_res                                  |
    */
    float reg[12], 
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    float3 calc_tmp, normalize_coef[4] = {
        make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0) };

    int dx, dy, load_N;
    uchar4 center[3][6], neigbour_window[3][6];
    float4 res;

    uint glo_dex = idx * pitch_src + idy;           *((float4*)&reg[0]) = src[glo_dex];
    glo_dex += 16 * pitch_src;                      *((float4*)&reg[4]) = src[glo_dex];

    store_to_shmem_L_uchar4;

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;       *((float4*)&reg[0]) = src[glo_dex];
        glo_dex += 16 * pitch_src;                  *((float4*)&reg[4]) = src[glo_dex];

        store_to_shmem_R_uchar4;
    }

    __syncthreads();

    // load values for center
    for (dx = 0; dx < 6; ++dx) {
        center[0][dx] = src_frag[threadIdx.x + 7][4 * threadIdx.y + 7 + dx];
        center[1][dx] = src_frag[threadIdx.x + 8][4 * threadIdx.y + 7 + dx];
        center[2][dx] = src_frag[threadIdx.x + 9][4 * threadIdx.y + 7 + dx];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 6; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][4 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][4 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][4 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_3x6_uchar4((float*)(&neigbour_window[0]));
            reg_shift_3x6_uchar4((float*)(&neigbour_window[1]));
            reg_shift_3x6_uchar4((float*)(&neigbour_window[2]));

            neigbour_window[0][5] = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 5];
            neigbour_window[1][5] = src_frag[threadIdx.x + dx + 1][4 * (threadIdx.y) + dy + 5];
            neigbour_window[2][5] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 5];
        }

        // element one
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][0], neigbour_window[load_N][0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[0], &normalize_coef[0], tot_div, &calc_tmp, &neigbour_window[1][1]);
        
        // element two
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[3], &normalize_coef[1], tot_div, &calc_tmp, &neigbour_window[1][2]);

        // element three
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[6], &normalize_coef[2], tot_div, &calc_tmp, &neigbour_window[1][3]);

        // element four
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 3
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[9], &normalize_coef[3], tot_div, &calc_tmp, &neigbour_window[1][4]);
    }

    cu_NLM_normalization_BGR_kpal(reg, normalize_coef, (uchar4*)&res, &center[1][1]);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*             \96 floats     16 uchar4s
* -----------------------------------
* |                                 |        16 uchar4s
* |         -----------------       |
* |         |               |       |
* |    apron|     constant  |       |        48 floats  => __shared__ uchar4 src_frag[32][80]
* |         |               |       |
* |         -----------------       |
* |                                 |
* -----------------------------------
*/
void cu_NLM_r8_BGR_KPAL_N5x5(float4*               src,
                             float4*               dst,
                             const uint            pitch_src,
                             const uint            pitch_dst,
                             const uint            eq_total_ker_len,
                             const uint            eq_Wker,
                             const uint2           eq_kernel_shift,
                             const float           h_2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ uchar4 src_frag[32][80 + 1];

    /**
    *             0   1   2   3   4   5   6   7   8   9  10  11
    *    time  [  x   x   x   x   x   x   x   x   x   x   x   x  ]
    *     1     |load from src  |    
    *     2     |  temp for a vec4 weights                       |
    *     3     |  denoised_res                                  |
    */
    float reg[12], 
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    float3 calc_tmp, normalize_coef[4] = {
        make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0), make_float3(0, 0, 0) };

    int dx, dy, load_N;
    uchar4 center[5][8], neigbour_window[5][8];
    float4 res;

    uint glo_dex = idx * pitch_src + idy;           *((float4*)&reg[0]) = src[glo_dex];
    glo_dex += 16 * pitch_src;                      *((float4*)&reg[4]) = src[glo_dex];

    store_to_shmem_L_uchar4;

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;       *((float4*)&reg[0]) = src[glo_dex];
        glo_dex += 16 * pitch_src;                  *((float4*)&reg[4]) = src[glo_dex];

        store_to_shmem_R_uchar4;
    }

    __syncthreads();

    // load values for center
    for (dx = 0; dx < 8; ++dx) {
        center[0][dx] = src_frag[threadIdx.x + 6][4 * threadIdx.y + 6 + dx];
        center[1][dx] = src_frag[threadIdx.x + 7][4 * threadIdx.y + 6 + dx];
        center[2][dx] = src_frag[threadIdx.x + 8][4 * threadIdx.y + 6 + dx];
        center[3][dx] = src_frag[threadIdx.x + 9][4 * threadIdx.y + 6 + dx];
        center[4][dx] = src_frag[threadIdx.x + 10][4 * threadIdx.y + 6 + dx];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 8; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][4 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][4 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][4 * threadIdx.y + dy + load_N];
                neigbour_window[3][load_N] = src_frag[threadIdx.x + dx + 3][4 * threadIdx.y + dy + load_N];
                neigbour_window[4][load_N] = src_frag[threadIdx.x + dx + 4][4 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_5x8_uchar4((float*)(&neigbour_window[0]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[1]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[2]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[3]));
            reg_shift_5x8_uchar4((float*)(&neigbour_window[4]));

            neigbour_window[0][7] = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 7];
            neigbour_window[1][7] = src_frag[threadIdx.x + dx + 1][4 * (threadIdx.y) + dy + 7];
            neigbour_window[2][7] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 7];
            neigbour_window[3][7] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 7];
            neigbour_window[4][7] = src_frag[threadIdx.x + dx + 2][4 * (threadIdx.y) + dy + 7];
        }

        // element one
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][0], neigbour_window[load_N][0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[0], &normalize_coef[0], tot_div, &calc_tmp, &neigbour_window[2][2]);
        
        // element two
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[3], &normalize_coef[1], tot_div, &calc_tmp, &neigbour_window[2][3]);

        // element three
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][6], neigbour_window[load_N][6], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[6], &normalize_coef[2], tot_div, &calc_tmp, &neigbour_window[2][4]);

        // element four
        calc_tmp = make_float3(0, 0, 0);
#pragma unroll 5
        for (load_N = 0; load_N < 5; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][6], neigbour_window[load_N][6], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][7], neigbour_window[load_N][7], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[9], &normalize_coef[3], tot_div, &calc_tmp, &neigbour_window[2][5]);
    }

    cu_NLM_normalization_BGR_kpal(reg, normalize_coef, (uchar4*)&res, &center[2][2]);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



#endif