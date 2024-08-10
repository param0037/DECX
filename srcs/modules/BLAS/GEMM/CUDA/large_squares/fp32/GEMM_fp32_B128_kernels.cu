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

#include "../GEMM_kernels.cuh"
#include "MMA_FP32.cuh"



__global__ void decx::blas::GPUK::
cu_GEMM_fp32_kernel_32_128_128(const float* __restrict A,   const float* __restrict B, 
                            float* __restrict dst,       const uint2 proc_dims_v1, 
                            const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                            const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 32 / 2;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 32 / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * 4 + blockIdx.y * 128;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_L_v1);

    __shared__ float4 _frag_A[32][128 / 4 + 1];
    __shared__ float4 _frag_B[32][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 2 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 32); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v2){
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1));
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j){
                _reg_aux._arrf[0] = _regsA[0]._arrf[j];
                _reg_aux._arrf[1] = _regsA[1]._arrf[j];
                _reg_aux._arrf[2] = _regsA[2]._arrf[j];
                _reg_aux._arrf[3] = _regsA[3]._arrf[j];
                _frag_A[loc_tid_Ax * 2 + j][loc_tid_Ay] = _reg_aux._vf;
            }
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + 64 + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + (k+64) * pitchA_v1));
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j){
                _reg_aux._arrf[0] = _regsA[0]._arrf[j];
                _reg_aux._arrf[1] = _regsA[1]._arrf[j];
                _reg_aux._arrf[2] = _regsA[2]._arrf[j];
                _reg_aux._arrf[3] = _regsA[3]._arrf[j];
                _frag_A[loc_tid_Ax * 2 + j][loc_tid_Ay + 16] = _reg_aux._vf;
            }
        }
        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[k]._vf = *((float4*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 32; ++_l)
        {
            _regsA[0]._vf = _frag_A[_l][threadIdx.y * 4];
            _regsA[1]._vf = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[2]._vf = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[3]._vf = _frag_A[_l][threadIdx.y * 4 + 3];

            _reg_aux._vf = _frag_B[_l][threadIdx.x];

            _MMA_FP32_1_4_16_(_regsA, _reg_aux, _accu);
        }

        _Lloc_A += 16;
        _Lloc_B += 32;

        dex_A += 32;
        dex_B += 32 * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 4 + tidy * pitchdst_v1 * 16;

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll 16
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp32_F_kernel_32_128_128(const float* __restrict A,       const float* __restrict B, 
                                 const float* __restrict C,       float* __restrict dst,       
                                 const float alpha,               const float beta, 
                                 const uint2 proc_dims_v1,        const uint32_t _L_v1,        
                                 const uint32_t pitchA_v1,        const uint32_t pitchB_v1,    
                                 const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 32 / 2;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 32 / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * 4 + blockIdx.y * 128;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_L_v1);

    __shared__ float4 _frag_A[32][128 / 4 + 1];
    __shared__ float4 _frag_B[32][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 2 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 32); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v2){
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1));
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j){
                _reg_aux._arrf[0] = _regsA[0]._arrf[j];
                _reg_aux._arrf[1] = _regsA[1]._arrf[j];
                _reg_aux._arrf[2] = _regsA[2]._arrf[j];
                _reg_aux._arrf[3] = _regsA[3]._arrf[j];
                _frag_A[loc_tid_Ax * 2 + j][loc_tid_Ay] = _reg_aux._vf;
            }
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + 64 + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + (k+64) * pitchA_v1));
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j){
                _reg_aux._arrf[0] = _regsA[0]._arrf[j];
                _reg_aux._arrf[1] = _regsA[1]._arrf[j];
                _reg_aux._arrf[2] = _regsA[2]._arrf[j];
                _reg_aux._arrf[3] = _regsA[3]._arrf[j];
                _frag_A[loc_tid_Ax * 2 + j][loc_tid_Ay + 16] = _reg_aux._vf;
            }
        }
        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[k]._vf = *((float4*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 32; ++_l)
        {
            _regsA[0]._vf = _frag_A[_l][threadIdx.y * 4];
            _regsA[1]._vf = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[2]._vf = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[3]._vf = _frag_A[_l][threadIdx.y * 4 + 3];

            _reg_aux._vf = _frag_B[_l][threadIdx.x];

            _MMA_FP32_1_4_16_(_regsA, _reg_aux, _accu);
        }

        _Lloc_A += 16;
        _Lloc_B += 32;

        dex_A += 32;
        dex_B += 32 * pitchB_v1;

        __syncthreads();
    }

#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf.x = __fmul_rn(_accu[k]._vf.x, alpha);
        _accu[k]._vf.y = __fmul_rn(_accu[k]._vf.y, alpha);
        _accu[k]._vf.z = __fmul_rn(_accu[k]._vf.z, alpha);
        _accu[k]._vf.w = __fmul_rn(_accu[k]._vf.w, alpha);
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 4 + tidy * pitchdst_v1 * 16;

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll 16
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy + k < proc_dims_v1.y) 
            {
                _reg_aux._vf = *((float4*)(dst + dex_dst + k * pitchdst_v1));
                _accu[k]._vf.x = __fmaf_rn(_reg_aux._vf.x, beta, _accu[k]._vf.x);
                _accu[k]._vf.y = __fmaf_rn(_reg_aux._vf.y, beta, _accu[k]._vf.y);
                _accu[k]._vf.z = __fmaf_rn(_reg_aux._vf.z, beta, _accu[k]._vf.z);
                _accu[k]._vf.w = __fmaf_rn(_reg_aux._vf.w, beta, _accu[k]._vf.w);

                *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
            }
        }
    }
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp32_kernel_16_128_128(const float* __restrict A,   const float* __restrict B, 
                               float* __restrict dst,       const uint2 proc_dims_v1, 
                               const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                               const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 16 / 2;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 16 / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * _LDG_HA_step + blockIdx.y * 128;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_L_v1);

    __shared__ float4 _frag_A[16][128 / 4 + 1];
    __shared__ float4 _frag_B[16][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 2 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 16); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v2){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1));
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j){
                _reg_aux._arrf[0] = _regsA[0]._arrf[j];
                _reg_aux._arrf[1] = _regsA[1]._arrf[j];
                _reg_aux._arrf[2] = _regsA[2]._arrf[j];
                _reg_aux._arrf[3] = _regsA[3]._arrf[j];
                _frag_A[loc_tid_Ax * 2 + j][loc_tid_Ay] = _reg_aux._vf;
            }
        }
        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[k]._vf = *((float4*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 16; ++_l)
        {
            _regsA[0]._vf = _frag_A[_l][threadIdx.y * 4];
            _regsA[1]._vf = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[2]._vf = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[3]._vf = _frag_A[_l][threadIdx.y * 4 + 3];

            _reg_aux._vf = _frag_B[_l][threadIdx.x];

            _MMA_FP32_1_4_16_(_regsA, _reg_aux, _accu);
        }

        _Lloc_A += 4;
        _Lloc_B += 16;

        dex_A += 16;
        dex_B += 16 * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 4 + tidy * pitchdst_v1 * 16;

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll 16
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp32_F_kernel_16_128_128(const float* __restrict A,   const float* __restrict B, 
                                 const float* __restrict C,   float* __restrict dst,       
                                 const float alpha,           const float beta, 
                                 const uint2 proc_dims_v1,    const uint32_t _L_v1,        
                                 const uint32_t pitchA_v1,    const uint32_t pitchB_v1,    
                                 const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 16 / 2;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 16 / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * _LDG_HA_step + blockIdx.y * 128;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_L_v1);

    __shared__ float4 _frag_A[16][128 / 4 + 1];
    __shared__ float4 _frag_B[16][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 2 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 16); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v2){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1));
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j){
                _reg_aux._arrf[0] = _regsA[0]._arrf[j];
                _reg_aux._arrf[1] = _regsA[1]._arrf[j];
                _reg_aux._arrf[2] = _regsA[2]._arrf[j];
                _reg_aux._arrf[3] = _regsA[3]._arrf[j];
                _frag_A[loc_tid_Ax * 2 + j][loc_tid_Ay] = _reg_aux._vf;
            }
        }
        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[k]._vf = *((float4*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 16; ++_l)
        {
            _regsA[0]._vf = _frag_A[_l][threadIdx.y * 4];
            _regsA[1]._vf = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[2]._vf = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[3]._vf = _frag_A[_l][threadIdx.y * 4 + 3];

            _reg_aux._vf = _frag_B[_l][threadIdx.x];

            _MMA_FP32_1_4_16_(_regsA, _reg_aux, _accu);
        }

        _Lloc_A += 4;
        _Lloc_B += 16;

        dex_A += 16;
        dex_B += 16 * pitchB_v1;

        __syncthreads();
    }

#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf.x = __fmul_rn(_accu[k]._vf.x, alpha);
        _accu[k]._vf.y = __fmul_rn(_accu[k]._vf.y, alpha);
        _accu[k]._vf.z = __fmul_rn(_accu[k]._vf.z, alpha);
        _accu[k]._vf.w = __fmul_rn(_accu[k]._vf.w, alpha);
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 4 + tidy * pitchdst_v1 * 16;

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll 16
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy + k < proc_dims_v1.y) 
            {
                _reg_aux._vf = *((float4*)(C + dex_dst + k * pitchdst_v1));
                _accu[k]._vf.x = __fmaf_rn(_reg_aux._vf.x, beta, _accu[k]._vf.x);
                _accu[k]._vf.y = __fmaf_rn(_reg_aux._vf.y, beta, _accu[k]._vf.y);
                _accu[k]._vf.z = __fmaf_rn(_reg_aux._vf.z, beta, _accu[k]._vf.z);
                _accu[k]._vf.w = __fmaf_rn(_reg_aux._vf.w, beta, _accu[k]._vf.w);

                *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
            }  
        }
    }
}



template<uint32_t L>
__global__ void decx::blas::GPUK::
cu_GEMM_fp32_kernel_128_128_T(const float* __restrict A,   const float* __restrict B, 
                              float* __restrict dst,       const uint2 proc_dims_v1, 
                              const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                              const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t tidx_A = threadIdx.x + blockIdx.y * blockDim.x;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t H_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.y, 4);

    __shared__ float4 _frag_A[L][128 / 4];
    __shared__ float4 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = threadIdx.y * _LDG_HA_step;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = tidx_A * 4 + pitchA_v1 * _Lloc_A;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, L); ++i)
    {
        // Load from A
        if (tidx_A < H_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_A + k < _L_v1) _regsA[k]._vf = *((float4*)(A + dex_A + k * pitchA_v1));
                _frag_A[threadIdx.y * _LDG_HA_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[k]._vf = *((float4*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._vf = _frag_A[_l][threadIdx.y * 4];
            _regsA[1]._vf = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[2]._vf = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[3]._vf = _frag_A[_l][threadIdx.y * 4 + 3];

            _reg_aux._vf = _frag_B[_l][threadIdx.x];

            _MMA_FP32_1_4_16_(_regsA, _reg_aux, _accu);
        }

        _Lloc_A += L;
        _Lloc_B += L;

        dex_A += L * pitchA_v1;
        dex_B += L * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 4 + tidy * pitchdst_v1 * 16;

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll 16
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
}

template __global__ void decx::blas::GPUK::cu_GEMM_fp32_kernel_128_128_T<32>(const float* __restrict, const float* __restrict, float* __restrict,
    const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);

template __global__ void decx::blas::GPUK::cu_GEMM_fp32_kernel_128_128_T<16>(const float* __restrict, const float* __restrict, float* __restrict,
    const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);

template __global__ void decx::blas::GPUK::cu_GEMM_fp32_kernel_128_128_T<8>(const float* __restrict, const float* __restrict, float* __restrict,
    const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);



template<uint32_t L>
__global__ void decx::blas::GPUK::
cu_GEMM_fp32_F_kernel_128_128_T(const float* __restrict A,      const float* __restrict B, 
                                const float* __restrict C,      float* __restrict dst,       
                                const float alpha,              const float beta, 
                                const uint2 proc_dims_v1,       const uint32_t _L_v1,        
                                const uint32_t pitchA_v1,       const uint32_t pitchB_v1,    
                                const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t tidx_A = threadIdx.x + blockIdx.y * blockDim.x;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t H_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.y, 4);

    __shared__ float4 _frag_A[L][128 / 4];
    __shared__ float4 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = threadIdx.y * _LDG_HA_step;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = tidx_A * 4 + pitchA_v1 * _Lloc_A;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, L); ++i)
    {
        // Load from A
        if (tidx_A < H_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_A + k < _L_v1) _regsA[k]._vf = *((float4*)(A + dex_A + k * pitchA_v1));
                _frag_A[threadIdx.y * _LDG_HA_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[k]._vf = *((float4*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = _regsA[k]._vf;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._vf = _frag_A[_l][threadIdx.y * 4];
            _regsA[1]._vf = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[2]._vf = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[3]._vf = _frag_A[_l][threadIdx.y * 4 + 3];

            _reg_aux._vf = _frag_B[_l][threadIdx.x];

            _MMA_FP32_1_4_16_(_regsA, _reg_aux, _accu);
        }

        _Lloc_A += L;
        _Lloc_B += L;

        dex_A += L * pitchA_v1;
        dex_B += L * pitchB_v1;

        __syncthreads();
    }

#pragma unroll 16
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf.x = __fmul_rn(_accu[k]._vf.x, alpha);
        _accu[k]._vf.y = __fmul_rn(_accu[k]._vf.y, alpha);
        _accu[k]._vf.z = __fmul_rn(_accu[k]._vf.z, alpha);
        _accu[k]._vf.w = __fmul_rn(_accu[k]._vf.w, alpha);
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 4 + tidy * pitchdst_v1 * 16;

    if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll 16
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy + k < proc_dims_v1.y) 
            {
                _reg_aux._vf = *((float4*)(C + dex_dst + k * pitchdst_v1));
                _accu[k]._vf.x = __fmaf_rn(_reg_aux._vf.x, beta, _accu[k]._vf.x);
                _accu[k]._vf.y = __fmaf_rn(_reg_aux._vf.y, beta, _accu[k]._vf.y);
                _accu[k]._vf.z = __fmaf_rn(_reg_aux._vf.z, beta, _accu[k]._vf.z);
                _accu[k]._vf.w = __fmaf_rn(_reg_aux._vf.w, beta, _accu[k]._vf.w);

                *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
            }
        }
    }
}

template __global__ void decx::blas::GPUK::cu_GEMM_fp32_F_kernel_128_128_T<32>(const float* __restrict, const float* __restrict, const float* __restrict, 
    float* __restrict, const float alpha, const float beta, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);

template __global__ void decx::blas::GPUK::cu_GEMM_fp32_F_kernel_128_128_T<16>(const float* __restrict, const float* __restrict, const float* __restrict, 
    float* __restrict, const float alpha, const float beta, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);

template __global__ void decx::blas::GPUK::cu_GEMM_fp32_F_kernel_128_128_T<8>(const float* __restrict, const float* __restrict, const float* __restrict, 
    float* __restrict, const float alpha, const float beta, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint32_t);

