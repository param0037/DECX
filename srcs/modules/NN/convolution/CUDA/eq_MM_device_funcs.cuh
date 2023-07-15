/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _EQ_MM_DEVICE_FUNCS_CUH_
#define _EQ_MM_DEVICE_FUNCS_CUH_

#include "../../../core/basic.h"
#include "../../../classes/classes_util.h"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"

/*
* In the equivalent GEMM kernel, whether the frag_B(name of the shared memory used to load from matrix B)
* is transposed. In fact, Nvidia© Nsight System proofs that the two different kernels(transposed or not)
* perform almost the same.
*/
// The undefined version is abandoned (08/04/2023)
#define _FRAG_B_TRANSPOSED_




#define _SET_ZERO_FLOAT4_1x4_(arr){                    \
    *((float4*)&arr[0]) = make_float4(0, 0, 0, 0);                \
    *((float4*)&arr[1]) = make_float4(0, 0, 0, 0);                \
    *((float4*)&arr[2]) = make_float4(0, 0, 0, 0);                \
    *((float4*)&arr[3]) = make_float4(0, 0, 0, 0);                \
}




#define _SET_ZERO_FLOAT4_1x8_(arr){                                     \
    *((float4*)&arr[0]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[1]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[2]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[3]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[4]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[5]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[6]) = decx::utils::vec4_set1_fp32(0);               \
    *((float4*)&arr[7]) = decx::utils::vec4_set1_fp32(0);               \
}




#define _SET_ZERO_FLOAT4_2x4_(arr){                                     \
    *((float4*)&arr[0][0]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[1][0]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[2][0]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[3][0]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[0][1]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[1][1]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[2][1]) = decx::utils::vec4_set1_fp32(0);            \
    *((float4*)&arr[3][1]) = decx::utils::vec4_set1_fp32(0);            \
}



#define _STORE_TO_DEST_(reg_sum_dex){    dst[dex_A] = reg_sum[reg_sum_dex];        dex_A += dst_store_bound; }

#define _STORE_TO_DEST_FP16_(reg_sum_dex){    dst[dex_A] = *((float4*)&reg_A[reg_sum_dex][0]);        dex_A += dst_store_bound; }


#ifdef _FRAG_B_TRANSPOSED_


#define _MM_ONE_ELEMENT_(sum_label, row, col, row_A_label){                                                               \
    reg_sum[row].sum_label = fmaf(reg_A[0].row_A_label, reg_B[col][0].x, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[1].row_A_label, reg_B[col][0].y, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[2].row_A_label, reg_B[col][0].z, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[3].row_A_label, reg_B[col][0].w, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[4].row_A_label, reg_B[col][1].x, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[5].row_A_label, reg_B[col][1].y, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[6].row_A_label, reg_B[col][1].z, reg_sum[row].sum_label);   \
    reg_sum[row].sum_label = fmaf(reg_A[7].row_A_label, reg_B[col][1].w, reg_sum[row].sum_label);   \
}



// row -> [0, 8)
#define _MM_ONE_ELEMENT_FP16_(row, col){                                                               \
    reg_sum[row][col] = __hfma2(make_half2(reg_A[0]._arrh[row], reg_A[1]._arrh[row]), reg_B[col]._arrh2[0], reg_sum[row][col]); \
    reg_sum[row][col] = __hfma2(make_half2(reg_A[2]._arrh[row], reg_A[3]._arrh[row]), reg_B[col]._arrh2[1], reg_sum[row][col]); \
    reg_sum[row][col] = __hfma2(make_half2(reg_A[4]._arrh[row], reg_A[5]._arrh[row]), reg_B[col]._arrh2[2], reg_sum[row][col]); \
    reg_sum[row][col] = __hfma2(make_half2(reg_A[6]._arrh[row], reg_A[7]._arrh[row]), reg_B[col]._arrh2[3], reg_sum[row][col]); \
}




// row -> [0, 8)
#define _MM_ONE_ELEMENT_FP16_ACCU_(row, col){                                                               \
    reg_sum[row][col] = fmaf(__half2float(reg_A[0]._arrh[row]), __half2float(reg_B[col]._arrh[0]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[1]._arrh[row]), __half2float(reg_B[col]._arrh[1]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[2]._arrh[row]), __half2float(reg_B[col]._arrh[2]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[3]._arrh[row]), __half2float(reg_B[col]._arrh[3]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[4]._arrh[row]), __half2float(reg_B[col]._arrh[4]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[5]._arrh[row]), __half2float(reg_B[col]._arrh[5]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[6]._arrh[row]), __half2float(reg_B[col]._arrh[6]), reg_sum[row][col]);       \
    reg_sum[row][col] = fmaf(__half2float(reg_A[7]._arrh[row]), __half2float(reg_B[col]._arrh[7]), reg_sum[row][col]);       \
}


#else


#define _MM_ONE_ELEMENT_(sum_label, A_shift, B_label){                                                                        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].x, reg_B[0].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].y, reg_B[1].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].z, reg_B[2].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].w, reg_B[3].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].x, reg_B[4].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].y, reg_B[5].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].z, reg_B[6].B_label, reg_sum[A_shift].sum_label);                    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].w, reg_B[7].B_label, reg_sum[A_shift].sum_label);                    \
}


#define _MM_4x8_ONE_ROW_(row) {                            \
    _MM_ONE_ELEMENT_(x, row, x);                        \
    _MM_ONE_ELEMENT_(y, row, y);                        \
    _MM_ONE_ELEMENT_(z, row, z);                        \
    _MM_ONE_ELEMENT_(w, row, w);                        \
}


#define _MM_4x4x8_ {                \
    _MM_4x8_ONE_ROW_(0);            \
    _MM_4x8_ONE_ROW_(1);            \
    _MM_4x8_ONE_ROW_(2);            \
    _MM_4x8_ONE_ROW_(3);            \
}

__device__
void _MM_4x4x8_fp32(float4 reg_A[4][2], float4 reg_B[8], float4 reg_sum[4])
{
    _MM_4x8_ONE_ROW_(0);
    _MM_4x8_ONE_ROW_(1);
    _MM_4x8_ONE_ROW_(2);
    _MM_4x8_ONE_ROW_(3);
}


#endif





namespace decx
{
    namespace conv {
        namespace GPUK
        {
            __device__ __inline__
            static void _MM_4x4x8_fp32(float4 reg_A[8], float4 reg_B[4][2], float4 reg_sum[4])
            {
                _MM_ONE_ELEMENT_(x, 0, 0, x);           _MM_ONE_ELEMENT_(y, 0, 1, x);
                _MM_ONE_ELEMENT_(z, 0, 2, x);           _MM_ONE_ELEMENT_(w, 0, 3, x);

                _MM_ONE_ELEMENT_(x, 1, 0, y);           _MM_ONE_ELEMENT_(y, 1, 1, y);
                _MM_ONE_ELEMENT_(z, 1, 2, y);           _MM_ONE_ELEMENT_(w, 1, 3, y);

                _MM_ONE_ELEMENT_(x, 2, 0, z);           _MM_ONE_ELEMENT_(y, 2, 1, z);
                _MM_ONE_ELEMENT_(z, 2, 2, z);           _MM_ONE_ELEMENT_(w, 2, 3, z);

                _MM_ONE_ELEMENT_(x, 3, 0, w);           _MM_ONE_ELEMENT_(y, 3, 1, w);
                _MM_ONE_ELEMENT_(z, 3, 2, w);           _MM_ONE_ELEMENT_(w, 3, 3, w);
            }

            __device__ __inline__
            static void _MM_8x8x8_fp16(decx::utils::_cuda_vec128 reg_A[8], 
                                       decx::utils::_cuda_vec128 reg_B[8], __half2 reg_sum[8][8])
            {
#if __ABOVE_SM_53
#pragma unroll 8
                for (int i = 0; i < 8; ++i) {
                    _MM_ONE_ELEMENT_FP16_(i, 0);
                    _MM_ONE_ELEMENT_FP16_(i, 1);
                    _MM_ONE_ELEMENT_FP16_(i, 2);
                    _MM_ONE_ELEMENT_FP16_(i, 3);
                    _MM_ONE_ELEMENT_FP16_(i, 4);
                    _MM_ONE_ELEMENT_FP16_(i, 5);
                    _MM_ONE_ELEMENT_FP16_(i, 6);
                    _MM_ONE_ELEMENT_FP16_(i, 7);
                }
#endif
            }


            __device__ __inline__
            static void _MM_8x8x8_fp16_accu(decx::utils::_cuda_vec128 reg_A[8],
                    decx::utils::_cuda_vec128 reg_B[8], float reg_sum[8][8])
            {
#if __ABOVE_SM_53
#pragma unroll 8
                for (int i = 0; i < 8; ++i) {
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 0);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 1);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 2);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 3);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 4);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 5);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 6);
                    _MM_ONE_ELEMENT_FP16_ACCU_(i, 7);
                }
#endif
            }


        }
    }
}




#endif