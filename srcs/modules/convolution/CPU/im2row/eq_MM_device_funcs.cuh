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

/*
* In the equivalent GEMM kernel, whether the frag_B(name of the shared memory used to load from matrix B)
* is transposed. In fact, Nvidia© Nsight System proofs that the two different kernels(transposed or not)
* perform almost the same.
*/
#define _FRAG_B_TRANSPOSED_




#define _SET_ZERO_FLOAT4_1x4_(arr){                    \
    *((float4*)&arr[0]) = make_float4(0, 0, 0, 0);                \
    *((float4*)&arr[1]) = make_float4(0, 0, 0, 0);                \
    *((float4*)&arr[2]) = make_float4(0, 0, 0, 0);                \
    *((float4*)&arr[3]) = make_float4(0, 0, 0, 0);                \
}




#define _SET_ZERO_FLOAT4_1x8_(arr){                    \
    arr[0] = make_float4(0, 0, 0, 0);                \
    arr[1] = make_float4(0, 0, 0, 0);                \
    arr[2] = make_float4(0, 0, 0, 0);                \
    arr[3] = make_float4(0, 0, 0, 0);                \
    arr[4] = make_float4(0, 0, 0, 0);                \
    arr[5] = make_float4(0, 0, 0, 0);                \
    arr[6] = make_float4(0, 0, 0, 0);                \
    arr[7] = make_float4(0, 0, 0, 0);                \
}



#define _SET_ZERO_FLOAT4_2x4_(arr){                    \
    *((float4*)&arr[0][0]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[1][0]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[2][0]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[3][0]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[0][1]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[1][1]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[2][1]) = make_float4(0, 0, 0, 0);            \
    *((float4*)&arr[3][1]) = make_float4(0, 0, 0, 0);            \
}


#define _STORE_TO_DEST_(reg_sum_dex){    dst[dex_A] = reg_sum[reg_sum_dex];        dex_A += dst_store_bound; }

#define _STORE_TO_DEST_FP16_(reg_sum_dex){    dst[dex_A] = *((float4*)&reg_A[reg_sum_dex][0]);        dex_A += dst_store_bound; }


#ifdef _FRAG_B_TRANSPOSED_


#define _MM_ONE_ELEMENT_(sum_label, A_shift, B_shift){                                                                \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].x, reg_B[2 * B_shift].x, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].y, reg_B[2 * B_shift].y, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].z, reg_B[2 * B_shift].z, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][0].w, reg_B[2 * B_shift].w, reg_sum[A_shift].sum_label);        \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].x, reg_B[2 * B_shift + 1].x, reg_sum[A_shift].sum_label);    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].y, reg_B[2 * B_shift + 1].y, reg_sum[A_shift].sum_label);    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].z, reg_B[2 * B_shift + 1].z, reg_sum[A_shift].sum_label);    \
    reg_sum[A_shift].sum_label = fmaf(reg_A[A_shift][1].w, reg_B[2 * B_shift + 1].w, reg_sum[A_shift].sum_label);    \
}


#define _MM_4x8_ONE_ROW_(row) {                        \
_MM_ONE_ELEMENT_(x, row, 0);                        \
_MM_ONE_ELEMENT_(y, row, 1);                        \
_MM_ONE_ELEMENT_(z, row, 2);                        \
_MM_ONE_ELEMENT_(w, row, 3);                        \
}



#define _MM_4x4x8_ {            \
_MM_4x8_ONE_ROW_(0);            \
_MM_4x8_ONE_ROW_(1);            \
_MM_4x8_ONE_ROW_(2);            \
_MM_4x8_ONE_ROW_(3);            \
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




#define _MM_4x4x8_FP16_ONE_ELE_(__row, __col, sum_offset) {       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][0].x, reg_B[__col][0].x, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][0].y, reg_B[__col][0].y, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][0].z, reg_B[__col][0].z, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][0].w, reg_B[__col][0].w, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][1].x, reg_B[__col][1].x, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][1].y, reg_B[__col][1].y, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][1].z, reg_B[__col][1].z, reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = __hfma2(reg_A[__row][1].w, reg_B[__col][1].w, reg_sum[__row + sum_offset][__col]);       \
}




#define _MM_4x4x8_FP16_ONE_ROW_(__row, sum_offset) {    \
_MM_4x4x8_FP16_ONE_ELE_(__row, 0, sum_offset);      \
_MM_4x4x8_FP16_ONE_ELE_(__row, 1, sum_offset);      \
_MM_4x4x8_FP16_ONE_ELE_(__row, 2, sum_offset);      \
_MM_4x4x8_FP16_ONE_ELE_(__row, 3, sum_offset);      \
}




#define _MM_4x4x8_FP16_ONE_ELE_ACCU_(__row, __col, sum_offset) {       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].x.x), __half2float(reg_B[__col][0].x.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].x.y), __half2float(reg_B[__col][0].x.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].y.x), __half2float(reg_B[__col][0].y.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].y.y), __half2float(reg_B[__col][0].y.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].z.x), __half2float(reg_B[__col][0].z.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].z.y), __half2float(reg_B[__col][0].z.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].w.x), __half2float(reg_B[__col][0].w.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][0].w.y), __half2float(reg_B[__col][0].w.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].x.x), __half2float(reg_B[__col][1].x.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].x.y), __half2float(reg_B[__col][1].x.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].y.x), __half2float(reg_B[__col][1].y.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].y.y), __half2float(reg_B[__col][1].y.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].z.x), __half2float(reg_B[__col][1].z.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].z.y), __half2float(reg_B[__col][1].z.y), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].w.x), __half2float(reg_B[__col][1].w.x), reg_sum[__row + sum_offset][__col]);       \
reg_sum[__row + sum_offset][__col] = fmaf(__half2float(reg_A[__row][1].w.y), __half2float(reg_B[__col][1].w.y), reg_sum[__row + sum_offset][__col]);       \
}


#define _MM_4x4x8_FP16_ONE_ROW_ACCU_(__row, sum_offset) {    \
_MM_4x4x8_FP16_ONE_ELE_ACCU_(__row, 0, sum_offset);      \
_MM_4x4x8_FP16_ONE_ELE_ACCU_(__row, 1, sum_offset);      \
_MM_4x4x8_FP16_ONE_ELE_ACCU_(__row, 2, sum_offset);      \
_MM_4x4x8_FP16_ONE_ELE_ACCU_(__row, 3, sum_offset);      \
}



namespace decx
{
    namespace conv {
        namespace GPUK
        {
            __device__ __inline__
            static void _MM_4x4x8_fp32(float4 reg_A[4][2], float4 reg_B[8], float4 reg_sum[4])
            {
                _MM_4x8_ONE_ROW_(0);
                _MM_4x8_ONE_ROW_(1);
                _MM_4x8_ONE_ROW_(2);
                _MM_4x8_ONE_ROW_(3);
            }

            template <int offset>
            __device__ __inline__
            static void _MM_4x4x8_fp16_accu(half2_8 reg_A[4][2], half2_8 reg_B[4][2], float reg_sum[8][4])
            {
#if __ABOVE_SM_53
                _MM_4x4x8_FP16_ONE_ROW_ACCU_(0, offset);
                _MM_4x4x8_FP16_ONE_ROW_ACCU_(1, offset);
                _MM_4x4x8_FP16_ONE_ROW_ACCU_(2, offset);
                _MM_4x4x8_FP16_ONE_ROW_ACCU_(3, offset);
#endif
            }

            template <int offset>
            __device__
            void _MM_4x4x8_fp16(half2_8 reg_A[4][2], half2_8 reg_B[4][2], half2 reg_sum[8][4])
            {
#if __ABOVE_SM_53
                _MM_4x4x8_FP16_ONE_ROW_(0, offset);
                _MM_4x4x8_FP16_ONE_ROW_(1, offset);
                _MM_4x4x8_FP16_ONE_ROW_(2, offset);
                _MM_4x4x8_FP16_ONE_ROW_(3, offset);
#endif
            }
        }
    }
}




#endif