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
decx::GPUK::cu_mat_m_vec_fp16_L1(const float4* __restrict    mat_src,
                                 const float4* __restrict    vec_src,
                                 float* __restrict           dst,
                                 const uint32_t              Wsrc_v8, 
                                 uint32_t                    Wdst_v1, 
                                 const uint2                 proc_dims)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    decx::utils::_cuda_vec128 _recv, _vec_vals;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = mat_src[LDG_dex];
        _vec_vals._vf = vec_src[tidx];

        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrs[i] = 0;
                _vec_vals._arrs[i] = 0;
            }
        }
    }

    _thread_sum = __fmul_rn(__half2float(_recv._arrh[0]), __half2float(_vec_vals._arrh[0]));
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[1]), __half2float(_vec_vals._arrh[1]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[2]), __half2float(_vec_vals._arrh[2]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[3]), __half2float(_vec_vals._arrh[3]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[4]), __half2float(_vec_vals._arrh[4]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[5]), __half2float(_vec_vals._arrh[5]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[6]), __half2float(_vec_vals._arrh[6]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[7]), __half2float(_vec_vals._arrh[7]), _thread_sum);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
#endif
}




__global__ void 
decx::GPUK::cu_vec_m_mat_fp16_L1(const __half* __restrict     vec_src,
                                 const float4* __restrict     mat_src,
                                 float4* __restrict           dst,
                                 const uint32_t               Wsrc_v8,
                                 uint32_t                     Wdst_v4,
                                 const uint2                  proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex_x = threadIdx.x + blockDim.x * blockIdx.x * 2 + threadIdx.y * blockDim.x;

    __shared__ float4 _workspace[8][32 + 1];
    __shared__ __half _vec_vals[8];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;
    float _vec_val = 0;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv._vf = mat_src[LDG_dex];
        __half tmp = vec_src[tidy];
        if (threadIdx.x == 0) { _vec_vals[threadIdx.y] = tmp; }
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp3._vd.x = ((double*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp3._vd.y = ((double*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];
    _vec_val = __half2float(_vec_vals[threadIdx.x % 8]);

    tmp1._arrf2[0] = __half22float2(tmp3._arrh2[0]);
    tmp1._arrf2[1] = __half22float2(tmp3._arrh2[1]);
    tmp2._arrf2[0] = __half22float2(tmp3._arrh2[2]);
    tmp2._arrf2[1] = __half22float2(tmp3._arrh2[3]);

    tmp1._vf = decx::utils::cuda::__float_mul4_1(tmp1._vf, _vec_val);
    tmp2._vf = decx::utils::cuda::__float_mul4_1(tmp2._vf, _vec_val);

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[0], &tmp3._arrf[0]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[1], &tmp3._arrf[1]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[2], &tmp3._arrf[2]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[3], &tmp3._arrf[3]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[0], &tmp4._arrf[0]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[1], &tmp4._arrf[1]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[2], &tmp4._arrf[2]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[3], &tmp4._arrf[3]);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        _workspace[0][threadIdx.y * 4 + threadIdx.x / 8] = tmp3._vf;
        _workspace[1][threadIdx.y * 4 + threadIdx.x / 8] = tmp4._vf;
    }

    __syncthreads();

    if (STG_dex_x < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && threadIdx.y < 2) {
        tmp1._vf = _workspace[threadIdx.y][threadIdx.x];
        dst[blockIdx.y * Wdst_v4 + STG_dex_x] = tmp1._vf;
    }
#endif
}



__global__ void 
decx::GPUK::cu_mat_m_vec_fp16_L2(const float4 * __restrict      mat_src,
                                 const float4* __restrict       vec_src,
                                 __half* __restrict             dst,
                                 const uint32_t                 Wsrc_v8, 
                                 uint32_t                       Wdst_v1, 
                                 const uint2                    proc_dims)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    decx::utils::_cuda_vec128 _recv, _vec_vals;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = mat_src[LDG_dex];
        _vec_vals._vf = vec_src[tidx];

        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrs[i] = 0;
                _vec_vals._arrs[i] = 0;
            }
        }
    }

    _thread_sum = __fmul_rn(__half2float(_recv._arrh[0]), __half2float(_vec_vals._arrh[0]));
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[1]), __half2float(_vec_vals._arrh[1]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[2]), __half2float(_vec_vals._arrh[2]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[3]), __half2float(_vec_vals._arrh[3]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[4]), __half2float(_vec_vals._arrh[4]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[5]), __half2float(_vec_vals._arrh[5]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[6]), __half2float(_vec_vals._arrh[6]), _thread_sum);
    _thread_sum = __fmaf_rn(__half2float(_recv._arrh[7]), __half2float(_vec_vals._arrh[7]), _thread_sum);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = __float2half_rn(_warp_reduce_res);
    }
#endif
}




__global__ void 
decx::GPUK::cu_vec_m_mat_fp16_L2(const __half* __restrict       vec_src,
                                 const float4 * __restrict      mat_src, 
                                 float4* __restrict             dst,
                                 const uint32_t                 Wsrc_v8, 
                                 uint32_t                       Wdst_v8, 
                                 const uint2                    proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;

    __shared__ float4 _workspace[8][32 + 1];
    __shared__ __half _vec_vals[8];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;
    float _vec_val = 0;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv._vf = mat_src[LDG_dex];
        __half tmp = vec_src[tidy];
        if (threadIdx.x == 0) { _vec_vals[threadIdx.y] = tmp; }
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;

    __syncthreads();

    tmp3._vf = _workspace[threadIdx.x % 8][threadIdx.y * 4 + threadIdx.x / 8];
    _vec_val = __half2float(_vec_vals[threadIdx.x % 8]);

    tmp1._arrf2[0] = __half22float2(tmp3._arrh2[0]);
    tmp1._arrf2[1] = __half22float2(tmp3._arrh2[1]);
    tmp2._arrf2[0] = __half22float2(tmp3._arrh2[2]);
    tmp2._arrf2[1] = __half22float2(tmp3._arrh2[3]);

    tmp1._vf = decx::utils::cuda::__float_mul4_1(tmp1._vf, _vec_val);
    tmp2._vf = decx::utils::cuda::__float_mul4_1(tmp2._vf, _vec_val);

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[0], &tmp3._arrf[0]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[1], &tmp3._arrf[1]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[2], &tmp3._arrf[2]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[3], &tmp3._arrf[3]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[0], &tmp4._arrf[0]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[1], &tmp4._arrf[1]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[2], &tmp4._arrf[2]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[3], &tmp4._arrf[3]);

    // Convert back to fp16 at the endig stage of the kernel
    tmp1._arrh2[0] = __float22half2_rn(tmp3._arrf2[0]);
    tmp1._arrh2[1] = __float22half2_rn(tmp3._arrf2[1]);
    tmp1._arrh2[2] = __float22half2_rn(tmp4._arrf2[0]);
    tmp1._arrh2[3] = __float22half2_rn(tmp4._arrf2[1]);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        _workspace[threadIdx.x % 8][threadIdx.y * 4 + threadIdx.x / 8] = tmp1._vf;
    }

    __syncthreads();

    tmp1._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && threadIdx.y == 0) {
        dst[blockIdx.y * Wdst_v8 + tidx] = tmp1._vf;
    }
#endif
}




__global__ void 
decx::GPUK::cu_mat_m_vec_fp16_L3(const float4* __restrict       mat_src,
                                 const float4* __restrict       vec_src, 
                                 __half* __restrict             dst,
                                 const uint32_t                 Wsrc_v8, 
                                 uint32_t                       Wdst_v1, 
                                 const uint2                    proc_dims)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex = Wdst_v1 * tidy + blockIdx.x;

    uint32_t proc_W_v8 = decx::utils::ceil<uint32_t>(proc_dims.x, 8);

    decx::utils::_cuda_vec128 _recv, _vec_vals;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    __half _thread_sum = __float2half_rn(0), _warp_reduce_res = __float2half_rn(0);

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv._vf = mat_src[LDG_dex];
        _vec_vals._vf = vec_src[tidx];

        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv._arrs[i] = 0;
                _vec_vals._arrs[i] = 0;
            }
        }
    }

    __half2 _tmp = __hmul2(_recv._arrh2[0], _vec_vals._arrh2[0]);
    _tmp = __hfma2(_recv._arrh2[1], _vec_vals._arrh2[1], _tmp);
    _tmp = __hfma2(_recv._arrh2[2], _vec_vals._arrh2[2], _tmp);
    _tmp = __hfma2(_recv._arrh2[3], _vec_vals._arrh2[3], _tmp);

    _thread_sum = __hadd(_tmp.x, _tmp.y);

    decx::reduce::GPUK::cu_warp_reduce<__half, 32>(__hadd, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
#endif
}




__global__ void 
decx::GPUK::cu_vec_m_mat_fp16_L3(const __half* __restrict       vec_src,
                                 const float4* __restrict       mat_src,
                                 float4* __restrict             dst,
                                 const uint32_t                 Wsrc_v8, 
                                 uint32_t                       Wdst_v8, 
                                 const uint2                    proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;

    __shared__ float4 _workspace[8][32 + 1];
    __shared__ __half _vec_vals[8];

    decx::utils::_cuda_vec128 _recv;
    _recv._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;
    __half _vec_val;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv._vf = mat_src[LDG_dex];
        __half tmp = vec_src[tidy];
        if (threadIdx.x == 0) { _vec_vals[threadIdx.y] = tmp; }
    }

    _workspace[threadIdx.y][threadIdx.x] = _recv._vf;
    
    __syncthreads();

    tmp1._vf = _workspace[threadIdx.x % 8][threadIdx.y * 4 + threadIdx.x / 8];
    _vec_val = __half2float(_vec_vals[threadIdx.x % 8]);

    __syncwarp(0xffffffff);

    tmp1 = decx::utils::cuda::__hmul_v8_1(tmp1, _vec_val);

    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp1._arrh2[0], &tmp2._arrh2[0]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp1._arrh2[1], &tmp2._arrh2[1]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp1._arrh2[2], &tmp2._arrh2[2]);
    decx::reduce::GPUK::cu_warp_reduce<__half2, 32, 4>(__hadd2, &tmp1._arrh2[3], &tmp2._arrh2[3]);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        _workspace[threadIdx.x % 8][threadIdx.y * 4 + threadIdx.x / 8] = tmp2._vf;
    }

    __syncthreads();

    tmp1._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && threadIdx.y == 0) {
        dst[blockIdx.y * Wdst_v8 + tidx] = tmp1._vf;
    }
#endif
}
