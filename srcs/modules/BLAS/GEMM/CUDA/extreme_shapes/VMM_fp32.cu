/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "VMM_kernels.cuh"


__global__ void
decx::GPUK::cu_vec_m_mat_fp32(const float* __restrict       vec_src,
                              const float4* __restrict      mat_src,
                              float4* __restrict            dst,
                              const uint32_t                Wsrc_v4,
                              uint32_t                      Wdst_v4,
                              const uint2                   proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v4 + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float2 tmp1, tmp2, tmp3, tmp4;
    float _vec_val;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && tidy < proc_dims_v1.y) {
        _recv._vf = mat_src[LDG_dex];
        _vec_val = vec_src[tidy];
    }
    
    _recv._vf.x = __fmul_rn(_recv._vf.x, _vec_val);
    _recv._vf.y = __fmul_rn(_recv._vf.y, _vec_val);
    _recv._vf.z = __fmul_rn(_recv._vf.z, _vec_val);
    _recv._vf.w = __fmul_rn(_recv._vf.w, _vec_val);

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp1 = ((float2*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2 = ((float2*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__float_add2, &tmp1, &tmp3);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(decx::utils::cuda::__float_add2, &tmp2, &tmp4);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((float2*)_workspace[0])[threadIdx.y * 4 + threadIdx.x / 8] = tmp3;
        ((float2*)_workspace[0])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4;
    }

    __syncthreads();

    _recv._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && threadIdx.y == 0) {
        dst[STG_dex] = _recv._vf;
    }
}



__global__ void 
decx::GPUK::cu_mat_m_vec_fp32(const float4* __restrict      mat_src,
                              const float4* __restrict      vec_src,
                              float* __restrict             dst,
                              const uint32_t                Wsrc_v4, 
                              uint32_t                      Wdst_v1, 
                              const uint2                   proc_dims)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v4 = decx::utils::ceil<uint32_t>(proc_dims.x, 4);

    decx::utils::_cuda_vec128 _recv, _vec_vals;
    _recv._vf = decx::utils::vec4_set1_fp32(0);
    _vec_vals._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv._vf = mat_src[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv._arrf[i] = 0.f;
            }
        }
        _vec_vals._vf = vec_src[tidx];
    }

    _thread_sum = __fmul_rn(_recv._vf.x, _vec_vals._vf.x);
    _thread_sum = __fmaf_rn(_recv._vf.y, _vec_vals._vf.y, _thread_sum);
    _thread_sum = __fmaf_rn(_recv._vf.z, _vec_vals._vf.z, _thread_sum);
    _thread_sum = __fmaf_rn(_recv._vf.w, _vec_vals._vf.w, _thread_sum);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}