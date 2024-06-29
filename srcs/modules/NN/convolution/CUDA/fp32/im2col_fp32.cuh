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


#ifndef _IM2COL_FP32_CUH_
#define _IM2COL_FP32_CUH_


#include "../../../../core/basic.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"


#define _IM2COL_D4N_FP32_BLOCK_X_ _CUDA_WARP_SIZE_ * 4
#define _IM2COL_D4N_FP32_BLOCK_Y_ 2

#define _IM2COL_D12_FP32_BLOCK_X_ _CUDA_WARP_SIZE_ * 6
#define _IM2COL_D12_FP32_BLOCK_Y_ 2

#define _IM2COL_GET_STG_BLOCKDIM_X_(_block_dim_x, _dpitch_v1) ((_block_dim_x / _dpitch_v1) * 4)
#define _IM2COL_GET_STG_BLOCKDIM_Y_(_block_dim_y) (_block_dim_y)


#define _MAX_IM2COL_BUF_SIZE_ 2048 * 2048 * 4


//float _reg1 = threadIdx.x, _reg2 = warpSize + threadIdx.x;
//float tmp;
//
//for (int i = 0; i < 5; ++i) {
//    tmp = __shfl_sync(0xffffffff, _reg2, i, 32);
//    _reg1 = __shfl_down_sync(0xffffffff, _reg1, 1, 32);
//    if (threadIdx.x == 31) {
//        _reg1 = tmp;
//    }
//
//    printf("%d, ", (int)_reg1);
//    if (threadIdx.x == 31) {
//        printf("\n");
//    }
//}


namespace decx
{
namespace nn {
    namespace GPUK 
    {
        __global__ void cu_im2col_DP4_NB_fp32(const float4* src, float4* dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);

        template <bool _bound_T, bool _bound_B>
        __global__ void cu_im2col_DP4_BC_fp32(const float4* __restrict src, float4* __restrict dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);


        __global__ void cu_im2col_DP8_NB_fp32(const float4* src, float2* dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);

        template <bool _bound_T, bool _bound_B>
        __global__ void cu_im2col_DP8_BC_fp32(const float4* __restrict src, float2* __restrict dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);


        __global__ void cu_im2col_DP12_NB_fp32(const float4* src, float2* dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);

        template <bool _bound_T, bool _bound_B>
        __global__ void cu_im2col_DP12_BC_fp32(const float4* __restrict src, float2* __restrict dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);


        __global__ void cu_im2col_DP16_NB_fp32(const float4* src, float* dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);

        template <bool _bound_T, bool _bound_B>
        __global__ void cu_im2col_DP16_BC_fp32(const float4* __restrict src, float* __restrict dst, const uint2 dst_dims, const uint3 kernel_dims,
            const uint2 strides, const uint32_t wpitch_dst, const uint32_t wpitch_src, const uint64_t im2col_buf_pitch_v1);
    }
}
}



#endif
