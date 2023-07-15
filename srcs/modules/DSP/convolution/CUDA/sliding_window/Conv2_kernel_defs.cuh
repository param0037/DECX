/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_KERNEL_DEFS_CUH_
#define _CONV2_KERNEL_DEFS_CUH_

#include "../../../../core/basic.h"
#include "../../../../classes/classes_util.h"
#include "../../conv_utils.h"


#define init_valuef 0.f
#define init_valueUint 0

__device__ __inline
void reg_shift_fp16(half2_8* tmp_reg_ptr)
{
#if __ABOVE_SM_53
#pragma unroll 7
    for (int i = 1; i < 8; ++i) {
        __half tmp = ((__half*)tmp_reg_ptr)[i];
        ((__half*)tmp_reg_ptr)[i - 1] = tmp;
    }
#endif
}



__device__ __inline
void reg_shift_f(float4* tmp_reg_ptr)
{
#pragma unroll 3
    for (int i = 1; i < 4; ++i) {
        float tmp = ((float*)tmp_reg_ptr)[i];
        ((float*)tmp_reg_ptr)[i - 1] = tmp;
    }
}



/**
* 传统二维滑窗卷积, traditional 2D convolution by shifting kernel
* 归并，将不同大小的卷积核离散到几个特定的卷积核大小优化核上进行
* 
* /////////////////////////////////////////////////////////////////////////////////////
*/


#define store_to_shmem_L {                                                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y] = reg_0.x;                                                        \
    src_frag[threadIdx.x][4 * threadIdx.y + 1] = reg_0.y;                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y + 2] = reg_0.z;                                                    \
    src_frag[threadIdx.x][4 * threadIdx.y + 3] = reg_0.w;                                                    \
    src_frag[16 + threadIdx.x][4 * threadIdx.y] = reg_1.x;                                                    \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 1] = reg_1.y;                                                \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 2] = reg_1.z;                                                \
    src_frag[16 + threadIdx.x][4 * threadIdx.y + 3] = reg_1.w;                                                \
}                                                                                                            \


#define hstore_to_shmem_L {                                                                                    \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y]) = ((float4*)&reg_0)->x;                                \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y + 2]) = ((float4*)&reg_0)->y;                            \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y + 4]) = ((float4*)&reg_0)->z;                            \
    *((float*)&src_frag[threadIdx.x][8 * threadIdx.y + 6]) = ((float4*)&reg_0)->w;                            \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y]) = ((float4*)&reg_1)->x;                            \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y + 2]) = ((float4*)&reg_1)->y;                        \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y + 4]) = ((float4*)&reg_1)->z;                        \
    *((float*)&src_frag[16 + threadIdx.x][8 * threadIdx.y + 6]) = ((float4*)&reg_1)->w;                        \
}                                                                                                            \



#define store_to_shmem_R {                                                                                    \
    src_frag[threadIdx.x][64 + 4 * threadIdx.y] = reg_0.x;                                                    \
    src_frag[threadIdx.x][65 + 4 * threadIdx.y] = reg_0.y;                                                    \
    src_frag[threadIdx.x][66 + 4 * threadIdx.y] = reg_0.z;                                                    \
    src_frag[threadIdx.x][67 + 4 * threadIdx.y] = reg_0.w;                                                    \
    src_frag[16 + threadIdx.x][64 + 4 * threadIdx.y] = reg_1.x;                                                \
    src_frag[16 + threadIdx.x][65 + 4 * threadIdx.y] = reg_1.y;                                                \
    src_frag[16 + threadIdx.x][66 + 4 * threadIdx.y] = reg_1.z;                                                \
    src_frag[16 + threadIdx.x][67 + 4 * threadIdx.y] = reg_1.w;                                                \
}                                                                                                            \


#define hstore_to_shmem_R {    \
    *((float*)&src_frag[threadIdx.x][128 + 8 * threadIdx.y]) = ((float4*)&reg_0)->x;                        \
    *((float*)&src_frag[threadIdx.x][130 + 8 * threadIdx.y]) = ((float4*)&reg_0)->y;                        \
    *((float*)&src_frag[threadIdx.x][132 + 8 * threadIdx.y]) = ((float4*)&reg_0)->z;                        \
    *((float*)&src_frag[threadIdx.x][134 + 8 * threadIdx.y]) = ((float4*)&reg_0)->w;                        \
    *((float*)&src_frag[16 + threadIdx.x][128 + 8 * threadIdx.y]) = ((float4*)&reg_1)->x;                    \
    *((float*)&src_frag[16 + threadIdx.x][130 + 8 * threadIdx.y]) = ((float4*)&reg_1)->y;                    \
    *((float*)&src_frag[16 + threadIdx.x][132 + 8 * threadIdx.y]) = ((float4*)&reg_1)->z;                    \
    *((float*)&src_frag[16 + threadIdx.x][134 + 8 * threadIdx.y]) = ((float4*)&reg_1)->w;                    \
}                                                                                                            \



#define Conv_fmaf {                                                                                            \
    reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);                                                                \
    reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);                                                                \
    reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);                                                                \
    reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);                                                                \
}                                                                                                            \



#define hstore_to_shmem_L3(offset_x) {                                                                        \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y]) = ((float4*)&reg_0)->x;                    \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y + 2]) = ((float4*)&reg_0)->y;                \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y + 4]) = ((float4*)&reg_0)->z;                \
    *((float*)&src_frag[offset_x + threadIdx.x][8 * threadIdx.y + 6]) = ((float4*)&reg_0)->w;                \
}                                                                                                            \



#define store_to_shmem_L3(offset_x) {                                                                        \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y] = reg_0.x;                                            \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 1] = reg_0.y;                                        \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 2] = reg_0.z;                                        \
    src_frag[offset_x + threadIdx.x][4 * threadIdx.y + 3] = reg_0.w;                                        \
}                                                                                                            \



#define store_to_shmem_R3(offset_x) {                                                                        \
    src_frag[offset_x + threadIdx.x][64 + 4 * threadIdx.y] = reg_0.x;                                        \
    src_frag[offset_x + threadIdx.x][65 + 4 * threadIdx.y] = reg_0.y;                                        \
    src_frag[offset_x + threadIdx.x][66 + 4 * threadIdx.y] = reg_0.z;                                        \
    src_frag[offset_x + threadIdx.x][67 + 4 * threadIdx.y] = reg_0.w;                                        \
}                                                                                                            \



#define hstore_to_shmem_R3(offset_x) {                                                                        \
    *((float*)&src_frag[offset_x + threadIdx.x][128 + 8 * threadIdx.y]) = ((float4*)&reg_0)->x;                \
    *((float*)&src_frag[offset_x + threadIdx.x][130 + 8 * threadIdx.y]) = ((float4*)&reg_0)->y;                \
    *((float*)&src_frag[offset_x + threadIdx.x][132 + 8 * threadIdx.y]) = ((float4*)&reg_0)->z;                \
    *((float*)&src_frag[offset_x + threadIdx.x][134 + 8 * threadIdx.y]) = ((float4*)&reg_0)->w;                \
}                                                                                                            \



#define sharedmem_offset 1        // to mitigate even prevent blank conflict


#endif