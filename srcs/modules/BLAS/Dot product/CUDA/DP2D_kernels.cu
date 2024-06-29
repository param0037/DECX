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


#include "DP_kernels.cuh"

// ======================================================== fp32 ========================================================

// ---------------------------------------------------------- h ----------------------------------------------------------
__global__ void 
decx::blas::GPUK::cu_block_dot2D_1way_h_fp32(const float4 * __restrict   A,
                                            const float4 * __restrict   B, 
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

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v4 && tidy < proc_dims.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
        if (tidx == proc_W_v4 - 1) {
            for (int i = 4 - (proc_W_v4 * 4 - proc_dims.x); i < 4; ++i) {
                _recv_A._arrf[i] = 0.f;
                _recv_B._arrf[i] = 0.f;
            }
        }
    }

    _thread_sum = __fmul_rn(_recv_A._vf.x, _recv_B._vf.x);
    _thread_sum = fmaf(_recv_A._vf.y, _recv_B._vf.y, _thread_sum);
    _thread_sum = fmaf(_recv_A._vf.z, _recv_B._vf.z, _thread_sum);
    _thread_sum = fmaf(_recv_A._vf.w, _recv_B._vf.w, _thread_sum);

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
}



// ---------------------------------------------------------- v ----------------------------------------------------------

__global__ void 
decx::blas::GPUK::cu_block_dot2D_1way_v_fp32(const float4 * __restrict    A,
                                            const float4 * __restrict    B,
                                            float4* __restrict           dst,
                                            const uint32_t               Wsrc_v4, 
                                            uint32_t                     Wdst_v4, 
                                            const uint2                  proc_dims_v1)
{
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v4 * tidy + tidx;
    uint64_t STG_dex = blockIdx.y * Wdst_v4 + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    float2 tmp1, tmp2, tmp3, tmp4;
    
    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly 
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && tidy < proc_dims_v1.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
    }

    _recv_A._arrf[0] = __fmul_rn(_recv_A._arrf[0], _recv_B._arrf[0]);
    _recv_A._arrf[1] = __fmul_rn(_recv_A._arrf[1], _recv_B._arrf[1]);
    _recv_A._arrf[2] = __fmul_rn(_recv_A._arrf[2], _recv_B._arrf[2]);
    _recv_A._arrf[3] = __fmul_rn(_recv_A._arrf[3], _recv_B._arrf[3]);

    _workspace[threadIdx.y][threadIdx.x] = _recv_A._vf;

    __syncthreads();

    tmp1 = ((float2*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp2 = ((float2*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(
        (decx::utils::cuda::cu_math_ops<double>*)&decx::utils::cuda::__float_add2, &tmp1, &tmp3);
    decx::reduce::GPUK::cu_warp_reduce<double, 32, 4>(
        (decx::utils::cuda::cu_math_ops<double>*)&decx::utils::cuda::__float_add2, &tmp2, &tmp4);

    __syncthreads();

    if (threadIdx.x % 8 == 0) {
        ((float2*)_workspace[0])[threadIdx.y * 4 + threadIdx.x / 8] = tmp3;
        ((float2*)_workspace[0])[32 + threadIdx.y * 4 + threadIdx.x / 8] = tmp4;
    }

    __syncthreads();
    
    _recv_B._vf = _workspace[threadIdx.y][threadIdx.x];

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4) && threadIdx.y == 0) {
        dst[STG_dex] = _recv_B._vf;
    }
}



// ======================================================== fp16 ========================================================

// ---------------------------------------------------------- h ----------------------------------------------------------

__global__ void 
decx::blas::GPUK::cu_block_dot2D_1way_h_fp16_L1(const float4 * __restrict   A, 
                                               const float4 * __restrict   B, 
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

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv_A._arrs[i] = 0;
                _recv_B._arrs[i] = 0;
            }
        }
    }

    __half2 _accu = __hmul2(_recv_A._arrh2[0], _recv_B._arrh2[0]);
    _accu = __hfma2(_recv_A._arrh2[1], _recv_B._arrh2[1], _accu);
    _accu = __hfma2(_recv_A._arrh2[2], _recv_B._arrh2[2], _accu);
    _accu = __hfma2(_recv_A._arrh2[3], _recv_B._arrh2[3], _accu);
    _thread_sum = __fadd_rn(__half2float(_accu.x), __half2float(_accu.y));

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
#endif
}



__global__ void 
decx::blas::GPUK::cu_block_dot2D_1way_h_fp16_L2(const float4 * __restrict   A,
                                         const float4 * __restrict   B,
                                         __half* __restrict          dst,
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

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    float _thread_sum = 0, _warp_reduce_res = 0;

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv_A._arrs[i] = 0;
                _recv_B._arrs[i] = 0;
            }
        }
    }

    __half2 _accu = __hmul2(_recv_A._arrh2[0], _recv_B._arrh2[0]);
    _accu = __hfma2(_recv_A._arrh2[1], _recv_B._arrh2[1], _accu);
    _accu = __hfma2(_recv_A._arrh2[2], _recv_B._arrh2[2], _accu);
    _accu = __hfma2(_recv_A._arrh2[3], _recv_B._arrh2[3], _accu);
    _thread_sum = __fadd_rn(__half2float(_accu.x), __half2float(_accu.y));

    decx::reduce::GPUK::cu_warp_reduce<float, 32>(__fadd_rn, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = __float2half_rn(_warp_reduce_res);
    }
#endif
}



__global__ void 
decx::blas::GPUK::cu_block_dot2D_1way_h_fp16_L3(const float4 * __restrict   A,
                                         const float4 * __restrict   B,
                                         __half* __restrict          dst,
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

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    __half _thread_sum = __float2half_rn(0), _warp_reduce_res = __float2half_rn(0);

    if (tidx < proc_W_v8 && tidy < proc_dims.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
        if (tidx == proc_W_v8 - 1) {
            for (int i = 8 - (proc_W_v8 * 8 - proc_dims.x); i < 8; ++i) {
                _recv_A._arrs[i] = 0;
                _recv_B._arrs[i] = 0;
            }
        }
    }

    __half2 _accu = __hmul2(_recv_A._arrh2[0], _recv_B._arrh2[0]);
    _accu = __hfma2(_recv_A._arrh2[1], _recv_B._arrh2[1], _accu);
    _accu = __hfma2(_recv_A._arrh2[2], _recv_B._arrh2[2], _accu);
    _accu = __hfma2(_recv_A._arrh2[3], _recv_B._arrh2[3], _accu);
    _thread_sum = __hadd(_accu.x, _accu.y);

    decx::reduce::GPUK::cu_warp_reduce<__half, 32>(__hadd, &_thread_sum, &_warp_reduce_res);

    if (threadIdx.x == 0 && tidy < proc_dims.y) {
        dst[STG_dex] = _warp_reduce_res;
    }
#endif
}

// ---------------------------------------------------------- v ----------------------------------------------------------



__global__ void 
decx::blas::GPUK::cu_block_dot2D_1way_v_fp16_L1(const float4 * __restrict   A,
                                         const float4 * __restrict   B,
                                         float4* __restrict          dst,
                                         const uint32_t              Wsrc_v8, 
                                         uint32_t                    Wdst_v4, 
                                         const uint2                 proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;
    uint64_t STG_dex_x = threadIdx.x + blockDim.x * blockIdx.x * 2 + threadIdx.y * blockDim.x;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
    }

    _recv_A._arrh2[0] = __hmul2(_recv_A._arrh2[0], _recv_B._arrh2[0]);
    _recv_A._arrh2[1] = __hmul2(_recv_A._arrh2[1], _recv_B._arrh2[1]);
    _recv_A._arrh2[2] = __hmul2(_recv_A._arrh2[2], _recv_B._arrh2[2]);
    _recv_A._arrh2[3] = __hmul2(_recv_A._arrh2[3], _recv_B._arrh2[3]);
    _workspace[threadIdx.y][threadIdx.x] = _recv_A._vf;

    __syncthreads();

    tmp3._vd.x = ((double*)_workspace[threadIdx.x % 8])[threadIdx.y * 4 + threadIdx.x / 8];
    tmp3._vd.y = ((double*)_workspace[threadIdx.x % 8])[32 + threadIdx.y * 4 + threadIdx.x / 8];

    tmp1._arrf2[0] = __half22float2(tmp3._arrh2[0]);
    tmp1._arrf2[1] = __half22float2(tmp3._arrh2[1]);
    tmp2._arrf2[0] = __half22float2(tmp3._arrh2[2]);
    tmp2._arrf2[1] = __half22float2(tmp3._arrh2[3]);

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
decx::blas::GPUK::cu_block_dot2D_1way_v_fp16_L2(const float4 * __restrict   A,
                                         const float4 * __restrict   B,
                                         float4* __restrict          dst,
                                         const uint32_t              Wsrc_v8, 
                                         uint32_t                    Wdst_v8, 
                                         const uint2                 proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
    }

    _recv_A._arrh2[0] = __hmul2(_recv_A._arrh2[0], _recv_B._arrh2[0]);
    _recv_A._arrh2[1] = __hmul2(_recv_A._arrh2[1], _recv_B._arrh2[1]);
    _recv_A._arrh2[2] = __hmul2(_recv_A._arrh2[2], _recv_B._arrh2[2]);
    _recv_A._arrh2[3] = __hmul2(_recv_A._arrh2[3], _recv_B._arrh2[3]);
    _workspace[threadIdx.y][threadIdx.x] = _recv_A._vf;

    __syncthreads();

    tmp3._vf = _workspace[threadIdx.x % 8][threadIdx.y * 4 + threadIdx.x / 8];

    tmp1._arrf2[0] = __half22float2(tmp3._arrh2[0]);
    tmp1._arrf2[1] = __half22float2(tmp3._arrh2[1]);
    tmp2._arrf2[0] = __half22float2(tmp3._arrh2[2]);
    tmp2._arrf2[1] = __half22float2(tmp3._arrh2[3]);

    __syncwarp(0xffffffff);

    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[0], &tmp3._arrf[0]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[1], &tmp3._arrf[1]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[2], &tmp3._arrf[2]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp1._arrf[3], &tmp3._arrf[3]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[0], &tmp4._arrf[0]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[1], &tmp4._arrf[1]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[2], &tmp4._arrf[2]);
    decx::reduce::GPUK::cu_warp_reduce<float, 32, 4>(__fadd_rn, &tmp2._arrf[3], &tmp4._arrf[3]);

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
decx::blas::GPUK::cu_block_dot2D_1way_v_fp16_L3(const float4 * __restrict   A,
                                         const float4 * __restrict   B,
                                         float4* __restrict          dst,
                                         const uint32_t              Wsrc_v8, 
                                         uint32_t                    Wdst_v8, 
                                         const uint2                 proc_dims_v1)
{
#if __ABOVE_SM_53
    uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t tidy = threadIdx.y + blockDim.y * blockIdx.y;

    uint64_t LDG_dex = Wsrc_v8 * tidy + tidx;

    __shared__ float4 _workspace[8][32 + 1];

    decx::utils::_cuda_vec128 _recv_A, _recv_B;
    _recv_A._vf = decx::utils::vec4_set1_fp32(0);
    _recv_B._vf = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec128 tmp1, tmp2, tmp3, tmp4;

    /**
    * No need to fill in zeros to the remaining spaces, since the loading
    * process goes all the way down vertically. The process stops at exactly
    * where the matrix ends.
    */
    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 8) && tidy < proc_dims_v1.y) {
        _recv_A._vf = A[LDG_dex];
        _recv_B._vf = B[LDG_dex];
    }

    _recv_A._arrh2[0] = __hmul2(_recv_A._arrh2[0], _recv_B._arrh2[0]);
    _recv_A._arrh2[1] = __hmul2(_recv_A._arrh2[1], _recv_B._arrh2[1]);
    _recv_A._arrh2[2] = __hmul2(_recv_A._arrh2[2], _recv_B._arrh2[2]);
    _recv_A._arrh2[3] = __hmul2(_recv_A._arrh2[3], _recv_B._arrh2[3]);
    _workspace[threadIdx.y][threadIdx.x] = _recv_A._vf;

    __syncthreads();

    tmp1._vf = _workspace[threadIdx.x % 8][threadIdx.y * 4 + threadIdx.x / 8];

    __syncwarp(0xffffffff);

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
