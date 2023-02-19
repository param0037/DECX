/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "constant_fill_kernels.cuh"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"


__global__ void
decx::bp::GPUK::cu_fill1D_constant_v128_b32(float4* src, const float4 val, const size_t len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < len) {
        src[tid] = val;
    }
}



__global__ void
decx::bp::GPUK::cu_fill1D_constant_v128_b32_end(float4* src, const float4 val, const float4 _end_val, const size_t len)
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
        
        decx::bp::GPUK::cu_fill1D_constant_v128_b32_end << <decx::utils::ceil<size_t>(fill_len_vec4, decx::cuP.prop.maxThreadsPerBlock), 
            decx::cuP.prop.maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)src, decx::utils::vec4_set1_fp32(val), _end_val, fill_len_vec4);
    }
    else {
        const size_t fill_len_vec4 = fill_len / 4;

        decx::bp::GPUK::cu_fill1D_constant_v128_b32 << <decx::utils::ceil<size_t>(fill_len_vec4, decx::cuP.prop.maxThreadsPerBlock),
            decx::cuP.prop.maxThreadsPerBlock, 0, S->get_raw_stream_ref() >> > (
            (float4*)src, decx::utils::vec4_set1_fp32(val), fill_len_vec4);
    }
}