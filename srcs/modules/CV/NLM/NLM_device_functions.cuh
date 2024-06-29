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


#ifndef _CUDA_NLM_DEVICE_FUNCTIONS_CUH_
#define _CUDA_NLM_DEVICE_FUNCTIONS_CUH_

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



#define store_to_shmem_L_uchar4 {                                                                                        \
    src_frag[threadIdx.x][4 * threadIdx.y] = *((uchar4*)&reg[0]);                                                        \
    src_frag[threadIdx.x][4 * threadIdx.y + 1] = *((uchar4*)&reg[1]);                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y + 2] = *((uchar4*)&reg[2]);                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y + 3] = *((uchar4*)&reg[3]);                                                    \
    src_frag[16 + threadIdx.x][4 * threadIdx.y] = *((uchar4*)&reg[4]);                                                   \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 1] = *((uchar4*)&reg[5]);                                               \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 2] = *((uchar4*)&reg[6]);                                               \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 3] = *((uchar4*)&reg[7]);                                               \
}



#define store_to_shmem_R_uchar4 {                                                                                        \
    src_frag[threadIdx.x][64 + 4 * threadIdx.y] = *((uchar4*)&reg[0]);                                                   \
    src_frag[threadIdx.x][65 + 4 * threadIdx.y] = *((uchar4*)&reg[1]);                                                   \
    src_frag[threadIdx.x][66 + 4 * threadIdx.y] = *((uchar4*)&reg[2]);                                                   \
    src_frag[threadIdx.x][67 + 4 * threadIdx.y] = *((uchar4*)&reg[3]);                                                   \
    src_frag[16 + threadIdx.x][64 + 4 * threadIdx.y] = *((uchar4*)&reg[4]);                                              \
    src_frag[16 + threadIdx.x][65 + 4 * threadIdx.y] = *((uchar4*)&reg[5]);                                              \
    src_frag[16 + threadIdx.x][66 + 4 * threadIdx.y] = *((uchar4*)&reg[6]);                                              \
    src_frag[16 + threadIdx.x][67 + 4 * threadIdx.y] = *((uchar4*)&reg[7]);                                              \
}



#define store_to_shmem_L3_uchar(offset_x) {                                                                                 \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][16 * threadIdx.y]) = *((uchar4*)&reg[0]);                                  \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][16 * threadIdx.y + 4]) = *((uchar4*)&reg[1]);                              \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][16 * threadIdx.y + 8]) = *((uchar4*)&reg[2]);                              \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][16 * threadIdx.y + 12]) = *((uchar4*)&reg[3]);                             \
}                                                                                                                           \



#define store_to_shmem_R3_uchar(offset_x) {                                                                                 \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][256 + 16 * threadIdx.y]) = *((uchar4*)&reg[0]);                            \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][260 + 16 * threadIdx.y]) = *((uchar4*)&reg[1]);                            \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][264 + 16 * threadIdx.y]) = *((uchar4*)&reg[2]);                            \
    *((uchar4*)&src_frag[offset_x + threadIdx.x][268 + 16 * threadIdx.y]) = *((uchar4*)&reg[3]);                            \
}                                                                



#define store_to_shmem_L_uchar {                                                                                            \
    *((uchar4*)&src_frag[threadIdx.x][16 * threadIdx.y]) = *((uchar4*)&reg[0]);                                             \
    *((uchar4*)&src_frag[threadIdx.x][16 * threadIdx.y + 4]) = *((uchar4*)&reg[1]);                                         \
    *((uchar4*)&src_frag[threadIdx.x][16 * threadIdx.y + 8]) = *((uchar4*)&reg[2]);                                         \
    *((uchar4*)&src_frag[threadIdx.x][16 * threadIdx.y + 12]) = *((uchar4*)&reg[3]);                                        \
    *((uchar4*)&src_frag[16 + threadIdx.x][16 * threadIdx.y]) = *((uchar4*)&reg[4]);                                        \
    *((uchar4*)&src_frag[16 + threadIdx.x][16 * threadIdx.y + 4]) = *((uchar4*)&reg[5]);                                    \
    *((uchar4*)&src_frag[16 + threadIdx.x][16 * threadIdx.y + 8]) = *((uchar4*)&reg[6]);                                    \
    *((uchar4*)&src_frag[16 + threadIdx.x][16 * threadIdx.y + 12]) = *((uchar4*)&reg[7]);                                   \
}



#define store_to_shmem_R_uchar {                                                                                            \
    *((uchar4*)&src_frag[threadIdx.x][256 + 16 * threadIdx.y]) = *((uchar4*)&reg[0]);                                       \
    *((uchar4*)&src_frag[threadIdx.x][260 + 16 * threadIdx.y]) = *((uchar4*)&reg[1]);                                       \
    *((uchar4*)&src_frag[threadIdx.x][264 + 16 * threadIdx.y]) = *((uchar4*)&reg[2]);                                       \
    *((uchar4*)&src_frag[threadIdx.x][268 + 16 * threadIdx.y]) = *((uchar4*)&reg[3]);                                       \
    *((uchar4*)&src_frag[16 + threadIdx.x][256 + 16 * threadIdx.y]) = *((uchar4*)&reg[4]);                                  \
    *((uchar4*)&src_frag[16 + threadIdx.x][260 + 16 * threadIdx.y]) = *((uchar4*)&reg[5]);                                  \
    *((uchar4*)&src_frag[16 + threadIdx.x][264 + 16 * threadIdx.y]) = *((uchar4*)&reg[6]);                                  \
    *((uchar4*)&src_frag[16 + threadIdx.x][268 + 16 * threadIdx.y]) = *((uchar4*)&reg[7]);                                  \
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
void cu_NLM_calc_neigbour_diff_gray(uchar __cent, uchar __neig, float* res_ptr)
{
    float tmp_diff;
    tmp_diff = __fsub_rn((float)__cent, (float)__neig);
    tmp_diff = __fmul_rn(tmp_diff, tmp_diff);
    *res_ptr = __fadd_rn(tmp_diff, *res_ptr);
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
void reg_shift_5x8_uchar4(float* tmp_reg_ptr)
{
    float tmp;
#pragma unroll 7
    for (int i = 1; i < 8; ++i) {
        tmp = ((float*)tmp_reg_ptr)[i];
        ((float*)tmp_reg_ptr)[i - 1] = tmp;
    }
}



__device__
void reg_shift_3x18_uchar(uchar* tmp_reg_ptr)
{
    uchar tmp;
#pragma unroll 17
    for (int i = 1; i < 18; ++i) {
        tmp = tmp_reg_ptr[i];
        tmp_reg_ptr[i - 1] = tmp;
    }
}



__device__
void reg_shift_5x20_uchar(uchar* tmp_reg_ptr)
{
    uchar tmp;
#pragma unroll 19
    for (int i = 1; i < 20; ++i) {
        tmp = tmp_reg_ptr[i];
        tmp_reg_ptr[i - 1] = tmp;
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



__device__ __inline__
void cu_NLM_integrateweights_gray(float* denoised_res, float* norm_coef, float tot_div,
    float* calc_tmp, uchar* src_val)
{
    *calc_tmp = __expf(__fdividef(*calc_tmp, tot_div));     // 0-Mean_square_difference 3x3

    *norm_coef = __fadd_rn(*norm_coef, *calc_tmp);

    *denoised_res = fmaf(*calc_tmp, (float)*src_val, *denoised_res);
}



__device__
void cu_NLM_normalization_BGR(float* denoised, float3 coef[4], uchar4* res)
{
    for (int i = 0; i < 4; ++i) {
        res[i].x = (uchar)(__fdividef(denoised[i * 3], coef[i].x));
        res[i].y = (uchar)(__fdividef(denoised[i * 3 + 1], coef[i].y));
        res[i].z = (uchar)(__fdividef(denoised[i * 3 + 2], coef[i].z));
        res[i].w = 255;
    }
}



__device__
void cu_NLM_normalization_BGR_kpal(float* denoised, float3 coef[4], uchar4* res, uchar4* origin_data)
{
    for (int i = 0; i < 4; ++i) {
        res[i].x = (uchar)(__fdividef(denoised[i * 3], coef[i].x));
        res[i].y = (uchar)(__fdividef(denoised[i * 3 + 1], coef[i].y));
        res[i].z = (uchar)(__fdividef(denoised[i * 3 + 2], coef[i].z));
        res[i].w = origin_data[i].w;
    }
}



__device__ __inline__
void cu_NLM_normalization_gray16(float* denoised, float* coef, uchar* res)
{
    for (int i = 0; i < 16; ++i) {
        res[i] = (uchar)(__fdividef(denoised[i], coef[i]));
    }
}


#endif