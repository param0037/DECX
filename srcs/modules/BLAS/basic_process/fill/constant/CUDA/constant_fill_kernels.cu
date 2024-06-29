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


#include "constant_fill_kernels.cuh"
#include "../../../../../core/utils/decx_cuda_vectypes_ops.cuh"


__global__ void
decx::bp::GPUK::cu_fill1D_constant_v128_b32(float4* __restrict src, const float4 val, const size_t len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < len) {
        src[tid] = val;
    }
}



__global__ void
decx::bp::GPUK::cu_fill1D_constant_v128_b32_end(float4* __restrict src, const float4 val, const float4 _end_val, const size_t len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < len) {
        src[tid] = tid == (len - 1) ? _end_val : val;
    }
}


void decx::bp::cu_fill1D_constant_v128_b32_caller(float* src, const float val, const size_t fill_len, decx::cuda_stream* S)
{
    if (fill_len % 4) {
        float4 _end_val = decx::utils::vec4_set1_fp32(0);
        for (int i = 0; i < (fill_len % 4); ++i) {
            ((float*)&_end_val)[i] = val;
        }

        const size_t fill_len_vec4 = decx::utils::ceil<size_t>(fill_len, 4);
        
        decx::bp::GPUK::cu_fill1D_constant_v128_b32_end << <decx::utils::ceil<size_t>(fill_len_vec4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), 
            decx::cuda::_get_cuda_prop().maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)src, decx::utils::vec4_set1_fp32(val), _end_val, fill_len_vec4);
    }
    else {
        const size_t fill_len_vec4 = fill_len / 4;

        decx::bp::GPUK::cu_fill1D_constant_v128_b32 << <decx::utils::ceil<size_t>(fill_len_vec4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock),
            decx::cuda::_get_cuda_prop().maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)src, decx::utils::vec4_set1_fp32(val), fill_len_vec4);
    }
}



void decx::bp::cu_fill1D_constant_v128_b64_caller(double* src, const double val, const size_t fill_len, decx::cuda_stream* S)
{
    float4 value_v4;
    *((double2*)&value_v4) = decx::utils::vec2_set1_fp64(val);

    if (fill_len % 2) {
        float4 _end_val = decx::utils::vec4_set1_fp32(0);
        for (int i = 0; i < (fill_len % 2); ++i) {
            ((double*)&_end_val)[i] = val;
        }

        const size_t fill_len_vec4 = decx::utils::ceil<size_t>(fill_len, 2);

        decx::bp::GPUK::cu_fill1D_constant_v128_b32_end << <decx::utils::ceil<size_t>(fill_len_vec4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock),
            decx::cuda::_get_cuda_prop().maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, value_v4, _end_val, fill_len_vec4);
    }
    else {
        const size_t fill_len_vec4 = fill_len / 2;

        decx::bp::GPUK::cu_fill1D_constant_v128_b32 << <decx::utils::ceil<size_t>(fill_len_vec4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock),
            decx::cuda::_get_cuda_prop().maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, value_v4, fill_len_vec4);
    }
}



// --------------------------------------------------------- 2D ----------------------------------------------------------

__global__ void
decx::bp::GPUK::cu_fill2D_constant_v128_b32(float4* src, const float4 val, const uint2 proc_dims, const uint Wsrc)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint tidy = threadIdx.y + blockDim.y * blockIdx.y;

    size_t dex = tidx * Wsrc + tidy;
    if (tidx < proc_dims.y && tidy < proc_dims.x) {
        src[dex] = val;
    }
}



__global__ void
decx::bp::GPUK::cu_fill2D_constant_v128_b32_LF(float4* src, const float4 val, const float4 _end_val, const uint2 proc_dims, const uint Wsrc)
{
    uint tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint tidy = threadIdx.y + blockDim.y * blockIdx.y;

    size_t dex = tidx * Wsrc + tidy;
    if (tidx < proc_dims.y && tidy < proc_dims.x) {
        src[dex] = tidy == (proc_dims.x - 1) ? _end_val : val;
    }
}



void decx::bp::cu_fill2D_constant_v128_b32_caller(float* src, const float val, const uint2 proc_dims, const uint Wsrc, decx::cuda_stream* S)
{
    if (proc_dims.x % 4) {
        float4 _end_val = decx::utils::vec4_set1_fp32(0);
        for (int i = 0; i < (proc_dims.x % 4); ++i) {
            ((float*)&_end_val)[i] = val;
        }

        const uint vec4_pWsrc = decx::utils::ceil<uint>(proc_dims.x, 4);
        dim3 grid(decx::utils::ceil<uint>(proc_dims.y, 16), decx::utils::ceil<uint>(vec4_pWsrc, 16));
        dim3 block(16, 16);

        decx::bp::GPUK::cu_fill2D_constant_v128_b32_LF << <grid, block, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, decx::utils::vec4_set1_fp32(val), _end_val, make_uint2(vec4_pWsrc, proc_dims.y), Wsrc / 4);
    }
    else {
        const uint vec4_pWsrc = proc_dims.x / 4;
        dim3 grid(decx::utils::ceil<uint>(proc_dims.y, 16), decx::utils::ceil<uint>(vec4_pWsrc, 16));
        dim3 block(16, 16);

        decx::bp::GPUK::cu_fill2D_constant_v128_b32 << <grid, block, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, decx::utils::vec4_set1_fp32(val), make_uint2(vec4_pWsrc, proc_dims.y), Wsrc / 4);
    }
}




void decx::bp::cu_fill2D_constant_v128_b64_caller(double* src, const double val, const uint2 proc_dims, const uint Wsrc, decx::cuda_stream* S)
{
    float4 value_vec4;
    *((double2*)&value_vec4) = decx::utils::vec2_set1_fp64(val);

    if (proc_dims.x % 2) {
        float4 _end_val = decx::utils::vec4_set1_fp32(0);
        for (int i = 0; i < (proc_dims.x % 2); ++i) {
            ((double*)&_end_val)[i] = val;
        }

        const uint vec4_pWsrc = decx::utils::ceil<uint>(proc_dims.x, 2);
        dim3 grid(decx::utils::ceil<uint>(proc_dims.y, 16), decx::utils::ceil<uint>(vec4_pWsrc, 16));
        dim3 block(16, 16);

        decx::bp::GPUK::cu_fill2D_constant_v128_b32_LF << <grid, block, 0, S->get_raw_stream_ref() >> > (
            (float4*)src, value_vec4, _end_val, make_uint2(vec4_pWsrc, proc_dims.y), Wsrc / 2);
    }
    else {
        const uint vec4_pWsrc = proc_dims.x / 2;
        dim3 grid(decx::utils::ceil<uint>(proc_dims.y, 16), decx::utils::ceil<uint>(vec4_pWsrc, 16));
        dim3 block(16, 16);

        decx::bp::GPUK::cu_fill2D_constant_v128_b32 << <grid, block, 0, S->get_raw_stream_ref() >> > (
            (float4*)src, value_vec4, make_uint2(vec4_pWsrc, proc_dims.y), Wsrc / 2);
    }
}