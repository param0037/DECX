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

#ifndef _ARITHMETIC_KERNEL_DEFS_CUH_
#define _ARITHMETIC_KERNEL_DEFS_CUH_


#include "../../../CUSV/decx_cuda_vectypes_ops.cuh"


#define _CUDA_OP_1D_VO_(kernel_name, vectype, exec)       \
__global__ void decx::GPUK::                                \
kernel_name(const float4* __restrict src,                   \
            float4* __restrict dst,                         \
            const uint64_t proc_len_v) {                    \
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;   \
    decx::utils::_cuda_vec128 recv, stg;                    \
    recv.vectype = src[tid];                                \
    exec;                                                   \
    dst[tid] = stg;                                         \
}                                                           \



#define _CUDA_OP_1D_VCO_(kernel_name, vectype, type_const, exec)       \
__global__ void decx::GPUK::                                \
kernel_name(const float4* __restrict src,                   \
            const type_const constant,                      \
            float4* __restrict dst,                         \
            const uint64_t proc_len_v) {                    \
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;   \
    decx::utils::_cuda_vec128 recv, stg;                    \
    recv.vectype = src[tid];                                \
    exec;                                                   \
    dst[tid] = stg.vectype;                                 \
}                                                           \


#define _CUDA_OP_1D_VVO_(kernel_name, vectype, exec)        \
__global__ void decx::GPUK::                                \
kernel_name(const float4* __restrict A,                     \
            const float4* __restrict B,                     \
            float4* __restrict dst,                         \
            const uint64_t proc_len_v) {                    \
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;   \
    decx::utils::_cuda_vec128 recv_A, recv_B, stg;          \
    recv_A.vectype = A[tid];                                \
    recv_B.vectype = B[tid];                                \
    exec;                                                   \
    dst[tid] = stg.vectype;                                 \
}                                                           \


#define _CUDA_OP_1D_VVO_CALLER_(caller, kernel, type_INOUT, vectype)    \
void decx::GPUK::caller(const type_INOUT* __restrict A,                 \
             const type_INOUT* __restrict B,                            \
             type_INOUT* __restrict dst,                                \
             const uint64_t proc_len_v,                                 \
             const uint32_t block,                                      \
             const uint32_t grid,                                       \
             decx::cuda_stream* S) {                                    \
    decx::GPUK::kernel<<<grid, block, 0, S->get_raw_stream_ref()>>>(    \
        (float4*)A, (float4*)B, (float4*)dst, proc_len_v);              \
}                                                                       \



#define _CUDA_OP_1D_VCO_CALLER_(caller, kernel, type_INOUT, type_const, vectype)    \
void decx::GPUK::caller(const type_INOUT* __restrict src,                 \
             const type_const constant,                                 \
             type_INOUT* __restrict dst,                                \
             const uint64_t proc_len_v,                                 \
             const uint32_t block,                                      \
             const uint32_t grid,                                       \
             decx::cuda_stream* S) {                                    \
    decx::GPUK::kernel<<<grid, block, 0, S->get_raw_stream_ref()>>>(    \
        (float4*)src, constant, (float4*)dst, proc_len_v);              \
}                                                                       \


#endif
