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


#ifndef _CUDA_NLM_GRAY_CUH_
#define _CUDA_NLM_GRAY_CUH_

#include "../../../common/basic.h"
#include "NLM_device_functions.cuh"


__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* ������ά�ȸպö�����8
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
void cu_NLM_r16_gray_N3x3(float4*               src,
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
    
    __shared__ uchar src_frag[48][288 + 4];

    float calc_tmp, reg[16], normalize_coef[16],
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    int dx, dy, load_N;
    uchar center[3][18], neigbour_window[3][18];
    int4 res;

    for (load_N = 0; load_N < 16; ++load_N)     
        normalize_coef[load_N] = 0;

    uint glo_dex = idx * pitch_src + idy;   *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar(0);

    glo_dex += 16 * pitch_src;              *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar(16);

    glo_dex += 16 * pitch_src;              *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar(32);

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;       *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar(0);

        glo_dex += 16 * pitch_src;                  *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar(16);

        glo_dex += 16 * pitch_src;                  *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar(32);
    }

    __syncthreads();

    // load values for center
    for (load_N = 0; load_N < 18; ++load_N) {
        center[0][load_N] = src_frag[threadIdx.x + 15][16 * threadIdx.y + 15 + load_N];
        center[1][load_N] = src_frag[threadIdx.x + 16][16 * threadIdx.y + 15 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 17][16 * threadIdx.y + 15 + load_N];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[12]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 18; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_3x18_uchar((uchar*)(&neigbour_window[0]));
            reg_shift_3x18_uchar((uchar*)(&neigbour_window[1]));
            reg_shift_3x18_uchar((uchar*)(&neigbour_window[2]));

            neigbour_window[0][17] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + 17];
            neigbour_window[1][17] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + 17];
            neigbour_window[2][17] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + 17];
        }

        for (load_N = 0; load_N < 16; ++load_N)
        {
            calc_tmp = 0;
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 0], neigbour_window[0][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 1], neigbour_window[0][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 2], neigbour_window[0][load_N + 2], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 0], neigbour_window[1][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 1], neigbour_window[1][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 2], neigbour_window[1][load_N + 2], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 0], neigbour_window[2][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 1], neigbour_window[2][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 2], neigbour_window[2][load_N + 2], &calc_tmp);

            cu_NLM_integrateweights_gray(&reg[load_N], &normalize_coef[load_N], tot_div, &calc_tmp, &neigbour_window[1][1 + load_N]);
        }
    }

    cu_NLM_normalization_gray16(&reg[0], &normalize_coef[0], (uchar*)&res);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* ������ά�ȸպö�����8
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
void cu_NLM_r16_gray_N5x5(float4*               src,
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
    
    __shared__ uchar src_frag[48][288 + 4];

    float calc_tmp, reg[16], normalize_coef[16],
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    int dx, dy, load_N;
    uchar center[5][20], neigbour_window[5][20];
    int4 res;

    for (load_N = 0; load_N < 16; ++load_N)     
        normalize_coef[load_N] = 0;

    uint glo_dex = idx * pitch_src + idy;   *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar(0);

    glo_dex += 16 * pitch_src;              *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar(16);

    glo_dex += 16 * pitch_src;              *((float4*)&reg[0]) = src[glo_dex];
    store_to_shmem_L3_uchar(32);

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;       *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar(0);

        glo_dex += 16 * pitch_src;                  *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar(16);

        glo_dex += 16 * pitch_src;                  *((float4*)&reg[0]) = src[glo_dex];
        store_to_shmem_R3_uchar(32);
    }

    __syncthreads();

    // load values for center
    for (load_N = 0; load_N < 20; ++load_N) {
        center[0][load_N] = src_frag[threadIdx.x + 14][16 * threadIdx.y + 14 + load_N];
        center[1][load_N] = src_frag[threadIdx.x + 15][16 * threadIdx.y + 14 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 16][16 * threadIdx.y + 14 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 17][16 * threadIdx.y + 14 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 18][16 * threadIdx.y + 14 + load_N];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[12]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 20; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + load_N];
                neigbour_window[3][load_N] = src_frag[threadIdx.x + dx + 3][16 * threadIdx.y + dy + load_N];
                neigbour_window[4][load_N] = src_frag[threadIdx.x + dx + 4][16 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[0]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[1]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[2]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[3]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[4]));

            neigbour_window[0][19] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + 19];
            neigbour_window[1][19] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + 19];
            neigbour_window[2][19] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + 19];
            neigbour_window[3][19] = src_frag[threadIdx.x + dx + 3][16 * threadIdx.y + dy + 19];
            neigbour_window[4][19] = src_frag[threadIdx.x + dx + 4][16 * threadIdx.y + dy + 19];
        }

        for (load_N = 0; load_N < 16; ++load_N)
        {
            calc_tmp = 0;
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 0], neigbour_window[0][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 1], neigbour_window[0][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 2], neigbour_window[0][load_N + 2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 3], neigbour_window[0][load_N + 3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 4], neigbour_window[0][load_N + 4], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 0], neigbour_window[1][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 1], neigbour_window[1][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 2], neigbour_window[1][load_N + 2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 3], neigbour_window[1][load_N + 3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 4], neigbour_window[1][load_N + 4], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 0], neigbour_window[2][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 1], neigbour_window[2][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 2], neigbour_window[2][load_N + 2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 3], neigbour_window[2][load_N + 3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 4], neigbour_window[2][load_N + 4], &calc_tmp);

            cu_NLM_integrateweights_gray(&reg[load_N], &normalize_coef[load_N], tot_div, &calc_tmp, &neigbour_window[1][1 + load_N]);
        }
    }

    cu_NLM_normalization_gray16(&reg[0], &normalize_coef[0], (uchar*)&res);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* ������ά�ȸպö�����8
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
void cu_NLM_r8_gray_N3x3(float4*               src,
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
    
    __shared__ uchar src_frag[32][272 + 4];

    float calc_tmp, reg[16], normalize_coef[16],
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    int dx, dy, load_N;
    uchar center[3][18], neigbour_window[3][18];
    int4 res;

    for (load_N = 0; load_N < 16; ++load_N)
        normalize_coef[load_N] = 0;

    uint glo_dex = idx * pitch_src + idy;               *((float4*)&reg[0]) = src[glo_dex];
    glo_dex += 16 * pitch_src;                          *((float4*)&reg[4]) = src[glo_dex];
    store_to_shmem_L_uchar;

    if (threadIdx.y < 1) {
        glo_dex = idx * pitch_src + idy + 16;           *((float4*)&reg[0]) = src[glo_dex];
        glo_dex += 16 * pitch_src;                      *((float4*)&reg[4]) = src[glo_dex];

        store_to_shmem_R_uchar;
    }

    __syncthreads();

    // load values for center
    for (load_N = 0; load_N < 18; ++load_N) {
        center[0][load_N] = src_frag[threadIdx.x + 7][16 * threadIdx.y + 7 + load_N];
        center[1][load_N] = src_frag[threadIdx.x + 8][16 * threadIdx.y + 7 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 9][16 * threadIdx.y + 7 + load_N];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[12]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 18; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_3x18_uchar((uchar*)(&neigbour_window[0]));
            reg_shift_3x18_uchar((uchar*)(&neigbour_window[1]));
            reg_shift_3x18_uchar((uchar*)(&neigbour_window[2]));

            neigbour_window[0][17] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + 17];
            neigbour_window[1][17] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + 17];
            neigbour_window[2][17] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + 17];
        }

        for (load_N = 0; load_N < 16; ++load_N)
        {
            calc_tmp = 0;
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 0], neigbour_window[0][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 1], neigbour_window[0][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 2], neigbour_window[0][load_N + 2], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 0], neigbour_window[1][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 1], neigbour_window[1][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 2], neigbour_window[1][load_N + 2], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 0], neigbour_window[2][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 1], neigbour_window[2][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 2], neigbour_window[2][load_N + 2], &calc_tmp);

            cu_NLM_integrateweights_gray(&reg[load_N], &normalize_coef[load_N], tot_div, &calc_tmp, &neigbour_window[1][1 + load_N]);
        }
    }

    cu_NLM_normalization_gray16(&reg[0], &normalize_coef[0], (uchar*)&res);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



__global__
/**
* One thread process 4 uchar4
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* ������ά�ȸպö�����8
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
void cu_NLM_r8_gray_N5x5(float4*               src,
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
    
    __shared__ uchar src_frag[32][272 + 4];

    float calc_tmp, reg[16], normalize_coef[16],
        tot_div = __fmul_rn(h_2, -9.f);     // = -h^2 * 9

    int dx, dy, load_N;
    uchar center[5][20], neigbour_window[5][20];
    int4 res;

    for (load_N = 0; load_N < 16; ++load_N)     
        normalize_coef[load_N] = 0;

    uint glo_dex = idx * pitch_src + idy;               *((float4*)&reg[0]) = src[glo_dex];
    glo_dex += 16 * pitch_src;                          *((float4*)&reg[4]) = src[glo_dex];
    store_to_shmem_L_uchar;

    if (threadIdx.y < 1) {
        glo_dex = idx * pitch_src + idy + 16;           *((float4*)&reg[0]) = src[glo_dex];
        glo_dex += 16 * pitch_src;                      *((float4*)&reg[4]) = src[glo_dex];

        store_to_shmem_R_uchar;
    }

    __syncthreads();

    // load values for center
    for (load_N = 0; load_N < 20; ++load_N) {
        center[0][load_N] = src_frag[threadIdx.x + 6][16 * threadIdx.y + 6 + load_N];
        center[1][load_N] = src_frag[threadIdx.x + 7][16 * threadIdx.y + 6 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 8][16 * threadIdx.y + 6 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 9][16 * threadIdx.y + 6 + load_N];
        center[2][load_N] = src_frag[threadIdx.x + 10][16 * threadIdx.y + 6 + load_N];
    }
    
    // set the registers all to zero
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[8]) = make_float4(0, 0, 0, 0);          *((float4*)&reg[12]) = make_float4(0, 0, 0, 0);

    for (int i = 0; i < eq_total_ker_len; ++i)
    {
        dx = eq_kernel_shift.x + i / eq_Wker;        dy = eq_kernel_shift.y + (i % eq_Wker);
        if (dy == eq_kernel_shift.y) {
            for (load_N = 0; load_N < 20; ++load_N) {
                neigbour_window[0][load_N] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + load_N];
                neigbour_window[1][load_N] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + load_N];
                neigbour_window[2][load_N] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + load_N];
                neigbour_window[3][load_N] = src_frag[threadIdx.x + dx + 3][16 * threadIdx.y + dy + load_N];
                neigbour_window[4][load_N] = src_frag[threadIdx.x + dx + 4][16 * threadIdx.y + dy + load_N];
            }
        }
        else {
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[0]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[1]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[2]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[3]));
            reg_shift_5x20_uchar((uchar*)(&neigbour_window[4]));

            neigbour_window[0][19] = src_frag[threadIdx.x + dx][16 * threadIdx.y + dy + 19];
            neigbour_window[1][19] = src_frag[threadIdx.x + dx + 1][16 * threadIdx.y + dy + 19];
            neigbour_window[2][19] = src_frag[threadIdx.x + dx + 2][16 * threadIdx.y + dy + 19];
            neigbour_window[3][19] = src_frag[threadIdx.x + dx + 3][16 * threadIdx.y + dy + 19];
            neigbour_window[4][19] = src_frag[threadIdx.x + dx + 4][16 * threadIdx.y + dy + 19];
        }

        for (load_N = 0; load_N < 16; ++load_N)
        {
            calc_tmp = 0;
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 0], neigbour_window[0][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 1], neigbour_window[0][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 2], neigbour_window[0][load_N + 2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 3], neigbour_window[0][load_N + 3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[0][load_N + 4], neigbour_window[0][load_N + 4], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 0], neigbour_window[1][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 1], neigbour_window[1][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 2], neigbour_window[1][load_N + 2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 3], neigbour_window[1][load_N + 3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[1][load_N + 4], neigbour_window[1][load_N + 4], &calc_tmp);

            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 0], neigbour_window[2][load_N + 0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 1], neigbour_window[2][load_N + 1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 2], neigbour_window[2][load_N + 2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 3], neigbour_window[2][load_N + 3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_gray(center[2][load_N + 4], neigbour_window[2][load_N + 4], &calc_tmp);

            cu_NLM_integrateweights_gray(&reg[load_N], &normalize_coef[load_N], tot_div, &calc_tmp, &neigbour_window[1][1 + load_N]);
        }
    }

    cu_NLM_normalization_gray16(&reg[0], &normalize_coef[0], (uchar*)&res);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}



#endif