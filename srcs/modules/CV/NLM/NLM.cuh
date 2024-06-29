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


#ifndef _CUDA_NLM_CUH_
#define _CUDA_NLM_CUH_

#include "../../core/basic.h"


#define store_to_shmem_L3_uchar4(offset_x) {                                                                \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y] = *((uchar4*)&reg[0]);                                \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 1] = *((uchar4*)&reg[1]);                            \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 2] = *((uchar4*)&reg[2]);                            \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 3] = *((uchar4*)&reg[3]);                            \
}                                                                                                           \



#define store_to_shmem_R3_uchar4(offset_x) {                                                                \
    src_frag[offset_x + threadIdx.x][64 + 4 * threadIdx.y] = *((uchar4*)&reg[0]);                           \
    src_frag[offset_x + threadIdx.x][65 + 4 * threadIdx.y] = *((uchar4*)&reg[1]);                           \
    src_frag[offset_x + threadIdx.x][66 + 4 * threadIdx.y] = *((uchar4*)&reg[2]);                           \
    src_frag[offset_x + threadIdx.x][67 + 4 * threadIdx.y] = *((uchar4*)&reg[3]);                           \
}                                                                        



__device__
void cu_NLM_calc_neigbour_diff_BGR(uchar4 __cent, uchar4 __neig, float3 *res_ptr)
{
    float tmp_diff;
    // B
    tmp_diff = __fsub_rn((float)__cent.x, (float)__neig.x);
    tmp_diff = __fmul_rn(tmp_diff, tmp_diff);
    res_ptr->x = __fadd_rn(tmp_diff, res_ptr->x);
    // G
    tmp_diff = __fsub_rn((float)__cent.y, (float)__neig.y);
    tmp_diff = __fmul_rn(tmp_diff, tmp_diff);
    res_ptr->y = __fadd_rn(tmp_diff, res_ptr->y);
    // R
    tmp_diff = __fsub_rn((float)__cent.z, (float)__neig.z);
    tmp_diff = __fmul_rn(tmp_diff, tmp_diff);
    res_ptr->z = __fadd_rn(tmp_diff, res_ptr->z);
}



__device__
void reg_shift_3x6_uchar4(float* tmp_reg_ptr)
{
    float tmp;
#pragma unroll 5
    for (int i = 1; i < 6; ++i) {
        tmp = ((float*)tmp_reg_ptr)[i];
        ((float*)tmp_reg_ptr)[i - 1] = tmp;
    }
}


__device__
void cu_NLM_integrateweights_BGR(float3* denoised_res, float3* norm_coef, float tot_div, 
    float3* calc_tmp, uchar4 *src_val)
{
    calc_tmp->x = __expf(__fdividef(calc_tmp->x, tot_div));     // B-Mean_square_difference 3x3
    calc_tmp->y = __expf(__fdividef(calc_tmp->y, tot_div));     // G-Mean_square_difference 3x3
    calc_tmp->z = __expf(__fdividef(calc_tmp->z, tot_div));     // R-Mean_square_difference 3x3

    norm_coef->x = __fadd_rn(norm_coef->x, calc_tmp->x);
    norm_coef->y = __fadd_rn(norm_coef->y, calc_tmp->y);
    norm_coef->z = __fadd_rn(norm_coef->z, calc_tmp->z);

    denoised_res->x = fmaf(calc_tmp->x, (float)src_val->x, denoised_res->x);
    denoised_res->y = fmaf(calc_tmp->y, (float)src_val->y, denoised_res->y);
    denoised_res->z = fmaf(calc_tmp->z, (float)src_val->z, denoised_res->z);
}


__device__
void cu_NLM_normalization(float* denoised, float3 coef[4], uchar4* res)
{
    for (int i = 0; i < 4; ++i) {
        res[i].x = (uchar)(__fdividef(denoised[i * 3], coef[i].x));
        res[i].y = (uchar)(__fdividef(denoised[i * 3 + 1], coef[i].y));
        res[i].z = (uchar)(__fdividef(denoised[i * 3 + 2], coef[i].z));
        res[i].w = 255;
        /*res[i].x = 255;
        res[i].y = 255;
        res[i].z = 255;
        res[i].w = 255;*/
    }
}


__global__
/**
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
void cu_NLM_BGR_N3x3_step1(float4*               src,
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
    *((float4*)&reg[0]) = make_float4(0, 0, 0, 0);
    *((float4*)&reg[4]) = make_float4(0, 0, 0, 0);
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
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][0], neigbour_window[load_N][0], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[0], &normalize_coef[0], tot_div, &calc_tmp, &neigbour_window[1][1]);
        
        // element two
        calc_tmp = make_float3(0, 0, 0);
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][1], neigbour_window[load_N][1], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[3], &normalize_coef[1], tot_div, &calc_tmp, &neigbour_window[1][2]);

        // element three
        calc_tmp = make_float3(0, 0, 0);
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][2], neigbour_window[load_N][2], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[6], &normalize_coef[2], tot_div, &calc_tmp, &neigbour_window[1][3]);

        // element four
        calc_tmp = make_float3(0, 0, 0);
        for (load_N = 0; load_N < 3; ++load_N) {
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][3], neigbour_window[load_N][3], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][4], neigbour_window[load_N][4], &calc_tmp);
            cu_NLM_calc_neigbour_diff_BGR(center[load_N][5], neigbour_window[load_N][5], &calc_tmp);
        }
        cu_NLM_integrateweights_BGR((float3*)&reg[9], &normalize_coef[3], tot_div, &calc_tmp, &neigbour_window[1][4]);
    }

    cu_NLM_normalization(reg, normalize_coef, (uchar4*)&res);

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = *((float4*)&res);
}

#endif