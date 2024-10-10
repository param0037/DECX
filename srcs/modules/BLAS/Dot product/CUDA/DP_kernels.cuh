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


#ifndef _DP_KERNELS_CUH_
#define _DP_KERNELS_CUH_


#include <Algorithms/reduce/CUDA/reduce_sum.cuh>


namespace decx
{
namespace blas
{
    namespace GPUK {
        /*
        * [32 * 8] (8 warps / block)
        */
        __global__ void cu_block_dot1D_fp32(const float4* __restrict A, const float4* __restrict B, float* __restrict dst,
            const uint64_t proc_len_v4, const uint64_t proc_len_v1);


        /*
        * [32 * 8] (8 warps / block)
        */
        __global__ void cu_block_dot1D_cplxf(const float4* __restrict A, const float4* __restrict B, de::CPf* __restrict dst,
            const uint64_t proc_len_v2, const uint64_t proc_len_v1);



        __global__ void cu_block_dot1D_fp64(const double2* __restrict A, const double2* __restrict B, double* __restrict dst,
            const uint64_t proc_len_v2, const uint64_t proc_len_v1);


        __global__ void cu_block_dot1D_fp16_L3(const float4* __restrict A, const float4* __restrict B, __half* __restrict dst,
            const uint64_t proc_len_v8, const uint64_t proc_len_v1);


        __global__ void cu_block_dot1D_fp16_L2(const float4* __restrict A, const float4* __restrict B, __half* __restrict dst,
            const uint64_t proc_len_v8, const uint64_t proc_len_v1);


        __global__ void cu_block_dot1D_fp16_L1(const float4* __restrict A, const float4* __restrict B, float* __restrict dst,
            const uint64_t proc_len_v8, const uint64_t proc_len_v1);
    }

    namespace GPUK 
    {
        // fp32
        __global__ void cu_block_dot2D_1way_h_fp32(const float4* __restrict A, const float4* __restrict B, float* __restrict dst, 
            const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


        __global__ void cu_block_dot2D_1way_v_fp32(const float4* __restrict A, const float4* __restrict B, float4* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v4, const uint2 proc_dims_v1);

        // fp16
        __global__ void cu_block_dot2D_1way_h_fp16_L1(const float4* __restrict A, const float4* __restrict B, float* __restrict dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


        __global__ void cu_block_dot2D_1way_h_fp16_L2(const float4* __restrict A, const float4* __restrict B, __half* __restrict dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


        __global__ void cu_block_dot2D_1way_h_fp16_L3(const float4* __restrict A, const float4* __restrict B, __half* __restrict dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


        __global__ void cu_block_dot2D_1way_v_fp16_L1(const float4* __restrict A, const float4* __restrict B, float4* __restrict dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v4, const uint2 proc_dims_v1);


        __global__ void cu_block_dot2D_1way_v_fp16_L2(const float4* __restrict A, const float4* __restrict B, float4* __restrict dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v8, const uint2 proc_dims_v1);


        __global__ void cu_block_dot2D_1way_v_fp16_L3(const float4* __restrict A, const float4* __restrict B, float4* __restrict dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v8, const uint2 proc_dims_v1);
    }
}
}


#endif