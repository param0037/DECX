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


#ifndef _VMM_KERNELS_CUH_
#define _VMM_KERNELS_CUH_


#include "../../../../core/basic.h"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../core/utils/decx_cuda_math_functions.cuh"
#include "../../../../basic_calculations/reduce/CUDA/reduce_sum.cuh"


namespace decx
{
namespace GPUK {
    __global__ void cu_vec_m_mat_fp32(const float* vec_src, const float4* mat_src, float4* dst,
        const uint32_t Wsrc_v4, uint32_t Wdst_v4, const uint2 proc_dims_v1);


    __global__ void cu_mat_m_vec_fp32(const float4* mat_src, const float4* vec_src, float* dst,
        const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


    /**
    * @brief All the calculations are done in fp32, all the way through. Don't need to worry about
    *        Overflow since only the very beggining loads fp16 to registers.
    */
    __global__ void cu_mat_m_vec_fp16_L1(const float4* mat_src, const float4* vec_src, float* dst,
        const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


    /**
    * @brief All the calculations are done in fp32, all the way through. Don't need to worry about
    *        Overflow since only the very beggining loads fp16 to registers.
    */
    __global__ void cu_vec_m_mat_fp16_L1(const half* vec_src, const float4* mat_src, float4* dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v4, const uint2 proc_dims_v1);


    /*
    * @brief All the calcualtions in the block is done in fp32, but communicate in fp16 between kernels
    *        (If necessary). Have to consider overflow, since when the number is huge in fp32, and converted
    *        to fp16 at the ending stage of each kernel, the overflow occurs.
    */
    __global__ void cu_mat_m_vec_fp16_L2(const float4* mat_src, const float4* vec_src, half* dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


    /*
    * @brief All the calcualtions in the block is done in fp32, but communicate in fp16 between kernels
    *        (If necessary). Have to consider overflow, since when the number is huge in fp32, and converted
    *        to fp16 at the ending stage of each kernel, the overflow occurs.
    */
    __global__ void cu_vec_m_mat_fp16_L2(const __half* vec_src, const float4* mat_src, float4* dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v8, const uint2 proc_dims_v1);


    /**
    * @brief All the calculations are done in fp16, all the way through. Have to worry about
    *        Overflow, making sure the maimum doesn't exceed max(fp16).
    */
    __global__ void cu_mat_m_vec_fp16_L3(const float4* mat_src, const float4* vec_src, __half* dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


    /**
    * @brief All the calculations are done in fp16, all the way through. Have to worry about
    *        Overflow, making sure the maimum doesn't exceed max(fp16).
    */
    __global__ void cu_vec_m_mat_fp16_L3(const __half* vec_src, const float4* mat_src, float4* dst,
            const uint32_t Wsrc_v8, uint32_t Wdst_v8, const uint2 proc_dims_v1);
}
}


#endif