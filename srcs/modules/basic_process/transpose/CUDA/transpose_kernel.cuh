/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TRANSPOSE_KERNEL_CUH_
#define _TRANSPOSE_KERNEL_CUH_


#include "../../../core/basic.h"
#include "../../../core/utils/decx_utils_device_functions.cuh"


#define _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ 16

namespace decx
{
    namespace bp {
        namespace GPUK {
            /**
            * Especailly for FFT2D, each thread process 4x4 de::CPf data (sizeof(de::CPf) = sizeof(double)), thus increasing the 
            * throughput
            */
            __global__ void cu_transpose_vec4x4d(double2* src, double2* dst, const uint pitchsrc,
                    const uint pitchdst, const uint2 proc_dim_dst);


            /**
            * Especailly for FFT2D, each thread process 4x4 de::CPf data (sizeof(de::CPf) = sizeof(double)), thus increasing the
            * throughput. Besides, this function also divide the de::CPf data during transposing, designed for IFFT2D
            */
            __global__ void cu_transpose_vec4x4d_and_divide(double2* src, double2* dst, const uint pitchsrc, const uint pitchdst,
                const float signal_len, const uint2 proc_dim_dst);


            /*
            * @param Wsrc : In float4, dev_tmp->width / 4
            * @param Wdst : In float4, dev_tmp->height / 4
            * @param proc_dims : the true dimension of source matrix, measured in element
            */
            __global__ void cu_transpose_vec4x4(const float4* src, float4* dst, const uint Wsrc, const uint Wdst, const uint2 proc_dims);


            /*
            * @param width : In float4, dev_tmp->width / 4
            * @param height : In float4, dev_tmp->height / 4
            * @param true_dim_src : the true dimension of source matrix, measured in element
            */
            __global__ void cu_transpose_vec8x8(const float4* src, float4* dst, const uint width, const uint height, const uint2 true_dim_src);


            /*
            * @param width : In float4, dev_tmp->width / 4
            * @param height : In float4, dev_tmp->height / 4
            */
            __global__ void cu_transpose_vec2x2(const double2* src, double2* dst, const uint width, const uint height, const uint2 proc_dims);
        }
    }
}


#endif
