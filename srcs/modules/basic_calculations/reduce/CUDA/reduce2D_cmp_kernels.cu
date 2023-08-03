/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "reduce_cmp.cuh"



template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32(const float4 * __restrict   src, 
                                                float* __restrict           dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v1, 
                                                const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    float _one_of_element;
    // The first thread of a warp load the value from the very beginning of the matrix of each row
    if (threadIdx.x == 0) {
        _one_of_element = ((float*)src)[Wsrc_v4 * tidy];
    }
    __syncwarp(0xffffffff);

    // Sync the data for the remaining 31 threads in a warp
    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(_one_of_element);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = _one_of_element;
            }
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::float4_max(_recv._vf);
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::float4_min(_recv._vf);
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}

template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<true>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<false>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);



template <bool _is_max>  __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp(const float4 * __restrict   src, 
                                                       float* __restrict           dst,
                                                       const uint32_t              Wsrc_v4, 
                                                       uint32_t                    Wdst_v1, 
                                                       const uint2                 proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * blockIdx.x + tidy;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    float _one_of_element;
    // The first thread of a warp load the value from the very beginning of the matrix of each row
    if (threadIdx.x == 0) {
        _one_of_element = ((float*)src)[Wsrc_v4 * tidy];
    }

    __syncwarp(0xffffffff);

    // Sync the data for the remaining 31 threads in a warp
    _one_of_element = __shfl_sync(0xffffffff, _one_of_element, 0, 32);

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(_one_of_element);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = _one_of_element;
            }
        }
    }

    if (_is_max) {
        _thread_sum = decx::reduce::GPUK::float4_max(_recv._vf);
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_max, &_thread_sum, &_warp_reduce_res);
    }
    else {
        _thread_sum = decx::reduce::GPUK::float4_min(_recv._vf);
        decx::reduce::GPUK::cu_warp_reduce_fp32<float(float, float), 32>(decx::utils::cuda::__fp32_min, &_thread_sum, &_warp_reduce_res);
    }

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}

template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp<true>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp<false>(const float4* __restrict, float* __restrict, const uint32_t, uint32_t, const uint2);




template <bool _is_max> __global__ void
decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32(const float4 * __restrict   src, 
                                                float4* __restrict          dst,
                                                const uint32_t              Wsrc_v4, 
                                                uint32_t                    Wdst_v4, 
                                                const uint2                 proc_dims_v4)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v4 + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    if (threadIdx.y == 0) {
        _recv._vf = src[(proc_dims_v4.y - 1) * Wsrc_v4];
        _workspace[0][threadIdx.x] = _recv._vf;
    }

    __syncthreads();
    _recv._vf = _workspace[0][threadIdx.x];
    __syncthreads();

    float2 tmp1, tmp2, tmp3, tmp4;
    
    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly 
    * where the matrix ends.
    */
    if (tidx < proc_dims_v4.x && tidy < proc_dims_v4.y) {
        _recv._vf = src[LDG_dex];
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp1 = ((float2*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2 = ((float2*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    if (_is_max) {
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float2_max, ((double*)&tmp1), ((double*)&tmp3));
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float2_max, ((double*)&tmp2), ((double*)&tmp4));
    }
    else {
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float2_min, ((double*)&tmp1), ((double*)&tmp3));
        decx::reduce::GPUK::cu_warp_reduce_fp64<double(double, double), 32, 4>(decx::utils::cuda::__float2_min, ((double*)&tmp2), ((double*)&tmp4));
    }

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((float2*)_workspace[0])[threadIdx.y * 4 + threadIdx.x / 8] = tmp3;
        ((float2*)_workspace[0])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4;
    }

    __syncthreads();
    
    _recv._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < proc_dims_v4.x && threadIdx.y == 0) {
        dst[STG_dex] = _recv._vf;
    }
}


template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32<true>(const float4* __restrict, float4* __restrict, const uint32_t, uint32_t, const uint2); 
template __global__ void decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32<false>(const float4* __restrict, float4* __restrict, const uint32_t, uint32_t, const uint2);