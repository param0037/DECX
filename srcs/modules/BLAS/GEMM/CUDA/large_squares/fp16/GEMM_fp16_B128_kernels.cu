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


/**
* Large matrix GEMM kernels on fp16 input don't provide any version other than 128x128.
* Since:
* 1. fp16 means computation capacitity > 560, such devices supports such amount of register
* 2. 128x128 which yields a 16x16 thread block, to use float4 load, this is the most
*    optimal approach.
* And, no fp16 accuracy level is enabled, only accumulates the results on fp32 accumulators.
* 1. To adapt tensor core technique.
* 2. Large matrices usually have large L, in general the results always overflow in fp16 mode.
*/

#include "../GEMM_kernels.cuh"


#if __ABOVE_SM_53
#define _DECL_A_DEX_OUTER_FP16_(_row_id) (_row_id >> 3)
#define _DECL_A_DEX_INNER_FP16_(_row_id) (_row_id & 7)
#define _DECL_A_ARG_FP16_(regsA, _row_id) (regsA[_DECL_A_DEX_OUTER_FP16_(_row_id)]._arrh[_DECL_A_DEX_INNER_FP16_(_row_id)])

#define _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, _row_id) {  \
    accu[_row_id]._v_h2_2[0] = __hfma2(__half2half2(_DECL_A_ARG_FP16_(regsA, _row_id)), regsB._v_h2_2[0], accu[_row_id]._v_h2_2[0]);     \
    accu[_row_id]._v_h2_2[1] = __hfma2(__half2half2(_DECL_A_ARG_FP16_(regsA, _row_id)), regsB._v_h2_2[1], accu[_row_id]._v_h2_2[1]);     \
}


#define _MMA_FP16_1_4_16_REG64_(regsA, regsB, accu) {   \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 0);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 1);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 2);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 3);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 4);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 5);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 6);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 7);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 8);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 9);      \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 10);     \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 11);     \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 12);     \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 13);     \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 14);     \
    _MMA_FP16_1_4_1_REG64_(regsA, regsB, accu, 15);     \
}
#endif


__global__ void decx::blas::GPUK::
cu_GEMM_fp16_kernel_32_128_128(const __half* __restrict A,   const __half* __restrict B, 
                               float* __restrict dst,       const uint2 proc_dims_v1, 
                               const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                               const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
#if __ABOVE_SM_53
    constexpr uint32_t L = 32;
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = threadIdx.x % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = threadIdx.x / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * _LDG_HA_step + blockIdx.y * 128;

    // Rearrange the 2D thread layout from 8x32 to 16x16 for LDG from B
    const uint32_t loc_tid_Bx = threadIdx.x % 32;
    const uint32_t loc_tid_By = threadIdx.x / 32;
    const uint32_t tid_Bx = loc_tid_Bx + blockIdx.x * 16;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v4 = decx::utils::ceil<uint32_t>(_L_v1, 4);
    
    __shared__ float2 _frag_A[L][128 / 4 + 1];
    __shared__ float2 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec64 _accu_fp16[16];
    decx::utils::_cuda_vec128 _regsA[_LDG_HA_step], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = loc_tid_By * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 4 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tid_Bx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 32); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k){
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1));
            }

#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _reg_aux._arrh[0] = _regsA[0]._arrh[k];
                _reg_aux._arrh[1] = _regsA[1]._arrh[k];
                _reg_aux._arrh[2] = _regsA[2]._arrh[k];
                _reg_aux._arrh[3] = _regsA[3]._arrh[k];
                _frag_A[loc_tid_Ax + k * 8][loc_tid_Ay] = _reg_aux._arrf2[0];
            }
        }
        // Load from B
        if (tid_Bx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k){
                _regsA[0]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(B + dex_B + k * pitchB_v1));
                _frag_B[loc_tid_By * _LDG_HB_step + k][loc_tid_Bx] = _regsA[0]._arrf2[0];
            }
        }
        
        __syncthreads();
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);
        }
#pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._arrf2[0] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4];
            _regsA[0]._arrf2[1] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4 + 1];
            _regsA[1]._arrf2[0] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4 + 2];
            _regsA[1]._arrf2[1] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4 + 3];
            _reg_aux._arrf2[0] = ((float2*)_frag_B[_l])[loc_tid_Bx];

            _MMA_FP16_1_4_16_REG64_(_regsA, _reg_aux._arr_reg64[0], _accu_fp16);
        }
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu[k]._vf.x = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[0]), _accu[k]._vf.x);
            _accu[k]._vf.y = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[1]), _accu[k]._vf.y);
            _accu[k]._vf.z = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[2]), _accu[k]._vf.z);
            _accu[k]._vf.w = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[3]), _accu[k]._vf.w);
        }
        _Lloc_A += L / 4;
        _Lloc_B += L;

        dex_A += L;
        dex_B += L * pitchB_v1;

        __syncthreads();
    }

    const uint32_t STG_tidx = (threadIdx.x % 32) + blockIdx.x * 32;
    const uint32_t STG_tidy = (threadIdx.x / 32) + blockIdx.y * 8;

    // Store the results to dst.
    const uint64_t dex_dst = STG_tidx * 4 + STG_tidy * pitchdst_v1 * 16;

    if (STG_tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4)) {
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            if (STG_tidy * 16 + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
#endif
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp16_F_kernel_32_128_128(const __half* __restrict A,   const __half* __restrict B, 
                                 const __half* __restrict C,   float* __restrict dst,       
                                 const __half alpha,           const __half beta,
                                 const uint2 proc_dims_v1,     const uint32_t _L_v1,        
                                 const uint32_t pitchA_v1,     const uint32_t pitchB_v1,    
                                 const uint32_t pitchdst_v1)
{
#if __ABOVE_SM_53
    constexpr uint32_t L = 32;
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = threadIdx.x % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = threadIdx.x / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * _LDG_HA_step + blockIdx.y * 128;

    // Rearrange the 2D thread layout from 8x32 to 16x16 for LDG from B
    const uint32_t loc_tid_Bx = threadIdx.x % 32;
    const uint32_t loc_tid_By = threadIdx.x / 32;
    const uint32_t tid_Bx = loc_tid_Bx + blockIdx.x * 16;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v4 = decx::utils::ceil<uint32_t>(_L_v1, 4);
    
    __shared__ float2 _frag_A[L][128 / 4 + 1];
    __shared__ float2 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec64 _accu_fp16[16];
    decx::utils::_cuda_vec128 _regsA[_LDG_HA_step], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = loc_tid_By * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 4 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tid_Bx * 4 + pitchB_v1 * _Lloc_B;

    const uint32_t STG_tidx = (threadIdx.x % 32) + blockIdx.x * 32;
    const uint32_t STG_tidy = (threadIdx.x / 32) + blockIdx.y * 8;

    // Store the results to C.
    const uint64_t dex_C = STG_tidx * 4 + STG_tidy * pitchB_v1 * 16;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
        _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);

        if (STG_tidy * 16 + k < proc_dims_v1.y) _accu_fp16[k]._vf2 = *((float2*)(C + dex_C + k * pitchB_v1));

        _accu_fp16[k]._v_h2_2[0] = __hmul2(__half2half2(beta), _accu_fp16[k]._v_h2_2[0]);
        _accu_fp16[k]._v_h2_2[1] = __hmul2(__half2half2(beta), _accu_fp16[k]._v_h2_2[1]);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 32); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k){
                _regsA[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1));
            }

#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _reg_aux._arrh[0] = _regsA[0]._arrh[k];
                _reg_aux._arrh[1] = _regsA[1]._arrh[k];
                _reg_aux._arrh[2] = _regsA[2]._arrh[k];
                _reg_aux._arrh[3] = _regsA[3]._arrh[k];
                _frag_A[loc_tid_Ax + k * 8][loc_tid_Ay] = _reg_aux._arrf2[0];
            }
        }
        // Load from B
        if (tid_Bx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k){
                _regsA[0]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(B + dex_B + k * pitchB_v1));
                _frag_B[loc_tid_By * _LDG_HB_step + k][loc_tid_Bx] = _regsA[0]._arrf2[0];
            }
        }
        
        __syncthreads();

        if (i > 0){
#pragma unroll
            for (uint32_t k = 0; k < 16; ++k){
                _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);
            }
        }
#pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._arrf2[0] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4];
            _regsA[0]._arrh2[0] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[0]);
            _regsA[0]._arrh2[1] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[1]);

            _regsA[0]._arrf2[1] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4 + 1];
            _regsA[0]._arrh2[2] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[2]);
            _regsA[0]._arrh2[3] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[3]);

            _regsA[1]._arrf2[0] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4 + 2];
            _regsA[1]._arrh2[0] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[0]);
            _regsA[1]._arrh2[1] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[1]);

            _regsA[1]._arrf2[1] = _frag_A[(_l & 3) * 8 + (_l >> 2)][loc_tid_By * 4 + 3];
            _regsA[1]._arrh2[2] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[2]);
            _regsA[1]._arrh2[3] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[3]);

            _reg_aux._arrf2[0] = ((float2*)_frag_B[_l])[loc_tid_Bx];

            _MMA_FP16_1_4_16_REG64_(_regsA, _reg_aux._arr_reg64[0], _accu_fp16);
        }
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu[k]._vf.x = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[0]), _accu[k]._vf.x);
            _accu[k]._vf.y = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[1]), _accu[k]._vf.y);
            _accu[k]._vf.z = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[2]), _accu[k]._vf.z);
            _accu[k]._vf.w = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[3]), _accu[k]._vf.w);
        }
        _Lloc_A += L / 4;
        _Lloc_B += L;

        dex_A += L;
        dex_B += L * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = STG_tidx * 4 + STG_tidy * pitchdst_v1 * 16;

    if (STG_tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4)) {
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            if (STG_tidy * 16 + k < proc_dims_v1.y) *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
#endif
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp16_kernel_64_128_128(const __half* __restrict A,   const __half* __restrict B, 
                               float* __restrict dst,       const uint2 proc_dims_v1, 
                               const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                               const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
#if __ABOVE_SM_53
    constexpr uint32_t L = 64;
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = threadIdx.x % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = threadIdx.x / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * _LDG_HA_step + blockIdx.y * 128;

    // Rearrange the 2D thread layout from 8x32 to 16x16 for LDG from B
    const uint32_t loc_tid_Bx = threadIdx.x % 32;
    const uint32_t loc_tid_By = threadIdx.x / 32;
    const uint32_t tid_Bx = loc_tid_Bx + blockIdx.x * 32;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v4 = decx::utils::ceil<uint32_t>(_L_v1, 4);
    
    __shared__ float4 _frag_A[L][128 / 8 + 1];
    __shared__ float2 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec64 _accu_fp16[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = loc_tid_By * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 4 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tid_Bx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 32); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k){
                _regsA[k >> 1]._arrf2[k & 1] = decx::utils::vec2_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k >> 1]._arrf2[k & 1] = *((float2*)(A + dex_A + k * pitchA_v1));
            }

#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _reg_aux._arrh[0] = _regsA[0]._arrh[k];
                _reg_aux._arrh[1] = _regsA[0]._arrh[k + 4];
                _reg_aux._arrh[2] = _regsA[1]._arrh[k];
                _reg_aux._arrh[3] = _regsA[1]._arrh[k + 4];
                _reg_aux._arrh[4] = _regsA[2]._arrh[k];
                _reg_aux._arrh[5] = _regsA[2]._arrh[k + 4];
                _reg_aux._arrh[6] = _regsA[3]._arrh[k];
                _reg_aux._arrh[7] = _regsA[3]._arrh[k + 4];
                _frag_A[loc_tid_Ax + k * 16][loc_tid_Ay] = _reg_aux._vf;
            }
        }
        // Load from B
        if (tid_Bx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k){
                _regsA[0]._arrf2[0] = decx::utils::vec2_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(B + dex_B + k * pitchB_v1));
                _frag_B[loc_tid_By * _LDG_HB_step + k][loc_tid_Bx] = _regsA[0]._arrf2[0];
            }
        }
        
        __syncthreads();

#pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);
        }
#pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._vf = _frag_A[(_l % 4) * 16 + (_l / 4)][loc_tid_By * 2];
            _regsA[1]._vf = _frag_A[(_l % 4) * 16 + (_l / 4)][loc_tid_By * 2 + 1];
            
            _reg_aux._arrf2[0] = ((float2*)_frag_B[_l])[loc_tid_Bx];

            _MMA_FP16_1_4_16_REG64_(_regsA, _reg_aux._arr_reg64[0], _accu_fp16);
        }
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu[k]._vf.x = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[0]), _accu[k]._vf.x);
            _accu[k]._vf.y = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[1]), _accu[k]._vf.y);
            _accu[k]._vf.z = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[2]), _accu[k]._vf.z);
            _accu[k]._vf.w = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[3]), _accu[k]._vf.w);
        }
        _Lloc_A += L / 4;
        _Lloc_B += L;

        dex_A += L;
        dex_B += L * pitchB_v1;

        __syncthreads();
    }

    const uint32_t STG_tidx = loc_tid_Bx + blockIdx.x * 32;
    const uint32_t STG_tidy = loc_tid_By + blockIdx.y * 8;

    // Store the results to dst.
    const uint64_t dex_dst = STG_tidx * 4 + STG_tidy * pitchdst_v1 * 16;

    if (STG_tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            if (STG_tidy * 16 + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
#endif
}




__global__ void decx::blas::GPUK::
cu_GEMM_fp16_F_kernel_64_128_128(const __half* __restrict A,   const __half* __restrict B, 
                                 const __half* __restrict C,   float* __restrict dst,
                                 const __half alpha,           const __half beta, 
                                 const uint2 proc_dims_v1,     const uint32_t _L_v1,
                                 const uint32_t pitchA_v1,     const uint32_t pitchB_v1,
                                 const uint32_t pitchdst_v1)
{
#if __ABOVE_SM_53
    constexpr uint32_t L = 64;
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = threadIdx.x % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = threadIdx.x / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay * _LDG_HA_step + blockIdx.y * 128;

    // Rearrange the 2D thread layout from 8x32 to 16x16 for LDG from B
    const uint32_t loc_tid_Bx = threadIdx.x % 32;
    const uint32_t loc_tid_By = threadIdx.x / 32;
    const uint32_t tid_Bx = loc_tid_Bx + blockIdx.x * 32;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t L_v4 = decx::utils::ceil<uint32_t>(_L_v1, 4);
    
    __shared__ float4 _frag_A[L][128 / 8 + 1];
    __shared__ float2 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec64 _accu_fp16[16];
    decx::utils::_cuda_vec128 _regsA[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = loc_tid_By * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A * 4 + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tid_Bx * 4 + pitchB_v1 * _Lloc_B;

    const uint32_t STG_tidx = loc_tid_Bx + blockIdx.x * 32;
    const uint32_t STG_tidy = loc_tid_By + blockIdx.y * 8;

    // Store the results to dst.
    const uint64_t dex_C = STG_tidx * 4 + STG_tidy * pitchB_v1 * 16;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);

        if (STG_tidy * 16 + k < proc_dims_v1.y) _accu_fp16[k]._vf2 = *((float2*)(C + dex_C + k * pitchB_v1));

        _accu_fp16[k]._v_h2_2[0] = __hmul2(__half2half2(beta), _accu_fp16[k]._v_h2_2[0]);
        _accu_fp16[k]._v_h2_2[1] = __hmul2(__half2half2(beta), _accu_fp16[k]._v_h2_2[1]);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 32); ++i)
    {
        // Load from A
        if (_Lloc_A < L_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k){
                _regsA[k >> 1]._arrf2[k & 1] = decx::utils::vec2_set1_fp32(0);
                if (tid_Ay + k < proc_dims_v1.y) _regsA[k >> 1]._arrf2[k & 1] = *((float2*)(A + dex_A + k * pitchA_v1));
            }

#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                _reg_aux._arrh[0] = _regsA[0]._arrh[k];
                _reg_aux._arrh[1] = _regsA[0]._arrh[k + 4];
                _reg_aux._arrh[2] = _regsA[1]._arrh[k];
                _reg_aux._arrh[3] = _regsA[1]._arrh[k + 4];
                _reg_aux._arrh[4] = _regsA[2]._arrh[k];
                _reg_aux._arrh[5] = _regsA[2]._arrh[k + 4];
                _reg_aux._arrh[6] = _regsA[3]._arrh[k];
                _reg_aux._arrh[7] = _regsA[3]._arrh[k + 4];
                _frag_A[loc_tid_Ax + k * 16][loc_tid_Ay] = _reg_aux._vf;
            }
        }
        // Load from B
        if (tid_Bx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k){
                _regsA[0]._arrf2[0] = decx::utils::vec2_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(B + dex_B + k * pitchB_v1));
                _frag_B[loc_tid_By * _LDG_HB_step + k][loc_tid_Bx] = _regsA[0]._arrf2[0];
            }
        }
        
        __syncthreads();

        if (i > 0){
#pragma unroll
            for (uint32_t k = 0; k < 16; ++k){
                _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);
            }
        }
#pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._vf = _frag_A[(_l % 4) * 16 + (_l / 4)][loc_tid_By * 2];
            _regsA[0]._arrh2[0] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[0]);
            _regsA[0]._arrh2[1] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[1]);
            _regsA[0]._arrh2[2] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[2]);
            _regsA[0]._arrh2[3] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[3]);

            _regsA[1]._vf = _frag_A[(_l % 4) * 16 + (_l / 4)][loc_tid_By * 2 + 1];
            _regsA[1]._arrh2[0] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[0]);
            _regsA[1]._arrh2[1] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[1]);
            _regsA[1]._arrh2[2] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[2]);
            _regsA[1]._arrh2[3] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[3]);
            
            _reg_aux._arrf2[0] = ((float2*)_frag_B[_l])[loc_tid_Bx];

            _MMA_FP16_1_4_16_REG64_(_regsA, _reg_aux._arr_reg64[0], _accu_fp16);
        }
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu[k]._vf.x = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[0]), _accu[k]._vf.x);
            _accu[k]._vf.y = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[1]), _accu[k]._vf.y);
            _accu[k]._vf.z = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[2]), _accu[k]._vf.z);
            _accu[k]._vf.w = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[3]), _accu[k]._vf.w);
        }
        _Lloc_A += L / 4;
        _Lloc_B += L;

        dex_A += L;
        dex_B += L * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = STG_tidx * 4 + STG_tidy * pitchdst_v1 * 16;

    if (STG_tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4))
    {
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            if (STG_tidy * 16 + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
#endif
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp16_kernel_64_128_128_T(const __half* __restrict A,   const __half* __restrict B, 
                               float* __restrict dst,       const uint2 proc_dims_v1, 
                               const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                               const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
#if __ABOVE_SM_53
    constexpr uint32_t L = 64;
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t tidx_A = threadIdx.x + blockIdx.y * blockDim.x;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t H_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.y, 4);

    __shared__ float2 _frag_A[L][128 / 4];
    __shared__ float2 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec64 _accu_fp16[16];
    decx::utils::_cuda_vec128 _regsA[2];
    decx::utils::_cuda_vec64 _reg_aux;

    uint32_t _Lloc_A = threadIdx.y;
    uint32_t _Lloc_B = threadIdx.y;
    
    uint64_t dex_A = tidx_A * 4 + pitchA_v1 * _Lloc_A;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, L); ++i)
    {
        // Load from A
        if (tidx_A < H_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k){
                _regsA[0]._arrf2[0] = decx::utils::vec2_set1_fp32(0);
                if (_Lloc_A + k * 8 < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1 * 8));
                _frag_A[threadIdx.y + k * 8][threadIdx.x] = _regsA[0]._arrf2[0];
            }
        }

        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k){
                _regsA[0]._arrf2[0] = decx::utils::vec2_set1_fp32(0);
                if (_Lloc_B + k * 8 < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(B + dex_B + k * pitchB_v1 * 8));
                _frag_B[threadIdx.y + k * 8][threadIdx.x] = _regsA[0]._arrf2[0];
            }
        }

        __syncthreads();

    #pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);
        }
    #pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._arrf2[0] = _frag_A[_l][threadIdx.y * 4];
            _regsA[0]._arrf2[1] = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[1]._arrf2[0] = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[1]._arrf2[1] = _frag_A[_l][threadIdx.y * 4 + 3];
            _reg_aux._vf2 = _frag_B[_l][threadIdx.x];

            _MMA_FP16_1_4_16_REG64_(_regsA, _reg_aux, _accu_fp16);
        }
    #pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu[k]._vf.x = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[0]), _accu[k]._vf.x);
            _accu[k]._vf.y = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[1]), _accu[k]._vf.y);
            _accu[k]._vf.z = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[2]), _accu[k]._vf.z);
            _accu[k]._vf.w = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[3]), _accu[k]._vf.w);
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
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy * 16 + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
#endif
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp16_F_kernel_64_128_128_T(const __half* __restrict A,   const __half* __restrict B, 
                                   const __half* __restrict C,   float* __restrict dst,       
                                   const __half alpha,           const __half beta,      
                                   const uint2 proc_dims_v1,     const uint32_t _L_v1,        
                                   const uint32_t pitchA_v1,     const uint32_t pitchB_v1,    
                                   const uint32_t pitchdst_v1)
{
#if __ABOVE_SM_53
    constexpr uint32_t L = 64;
    constexpr uint32_t _loc_LDG_Ax = L / 4;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = L / 8;
    constexpr uint32_t _LDG_HA_step = 128 / _loc_LDG_Ay;

    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t tidx_A = threadIdx.x + blockIdx.y * blockDim.x;

    const uint32_t W_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4);
    const uint32_t H_v4 = decx::utils::ceil<uint32_t>(proc_dims_v1.y, 4);

    __shared__ float2 _frag_A[L][128 / 4];
    __shared__ float2 _frag_B[L][128 / 4];

    decx::utils::_cuda_vec128 _accu[16];
    decx::utils::_cuda_vec64 _accu_fp16[16];
    decx::utils::_cuda_vec128 _regsA[2];
    decx::utils::_cuda_vec64 _reg_aux;

    uint32_t _Lloc_A = threadIdx.y;
    uint32_t _Lloc_B = threadIdx.y;
    
    uint64_t dex_A = tidx_A * 4 + pitchA_v1 * _Lloc_A;
    uint64_t dex_B = tidx * 4 + pitchB_v1 * _Lloc_B;

    // Store the results to C.
    const uint64_t dex_C = tidx * 4 + tidy * pitchB_v1 * 16;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 16; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);

        if (tidx < decx::utils::ceil<uint32_t>(proc_dims_v1.x, 4)) {
            if (tidy * 16 + k < proc_dims_v1.y)  _accu_fp16[k]._vf2 = *((float2*)(C + dex_C + k * pitchB_v1));
        }

        _accu_fp16[k]._v_h2_2[0] = __hmul2(__half2half2(beta), _accu_fp16[k]._v_h2_2[0]);
        _accu_fp16[k]._v_h2_2[1] = __hmul2(__half2half2(beta), _accu_fp16[k]._v_h2_2[1]);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, L); ++i)
    {
        // Load from A
        if (tidx_A < H_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HA_step; ++k){
                _regsA[0]._arrf2[0] = decx::utils::vec2_set1_fp32(0);
                if (_Lloc_A + k * 8 < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(A + dex_A + k * pitchA_v1 * 8));
                _frag_A[threadIdx.y + k * 8][threadIdx.x] = _regsA[0]._arrf2[0];
            }
        }

        // Load from B
        if (tidx < W_v4){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k){
                _regsA[0]._arrf2[0] = decx::utils::vec2_set1_fp32(0);
                if (_Lloc_B + k * 8 < _L_v1) _regsA[0]._arrf2[0] = *((float2*)(B + dex_B + k * pitchB_v1 * 8));
                _frag_B[threadIdx.y + k * 8][threadIdx.x] = _regsA[0]._arrf2[0];
            }
        }

        __syncthreads();

        if (i > 0){
    #pragma unroll
            for (uint32_t k = 0; k < 16; ++k){
                _accu_fp16[k]._vf2 = decx::utils::vec2_set1_fp32(0);
            }
        }
    #pragma unroll
        for (uint32_t _l = 0; _l < L; ++_l)
        {
            _regsA[0]._arrf2[0] = _frag_A[_l][threadIdx.y * 4];
            _regsA[0]._arrh2[0] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[0]);
            _regsA[0]._arrh2[1] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[1]);

            _regsA[0]._arrf2[1] = _frag_A[_l][threadIdx.y * 4 + 1];
            _regsA[0]._arrh2[2] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[2]);
            _regsA[0]._arrh2[3] = __hmul2(__half2half2(alpha), _regsA[0]._arrh2[3]);

            _regsA[1]._arrf2[0] = _frag_A[_l][threadIdx.y * 4 + 2];
            _regsA[1]._arrh2[0] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[0]);
            _regsA[1]._arrh2[1] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[1]);
            
            _regsA[1]._arrf2[1] = _frag_A[_l][threadIdx.y * 4 + 3];
            _regsA[1]._arrh2[2] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[2]);
            _regsA[1]._arrh2[3] = __hmul2(__half2half2(alpha), _regsA[1]._arrh2[3]);

            _reg_aux._vf2 = _frag_B[_l][threadIdx.x];

            _MMA_FP16_1_4_16_REG64_(_regsA, _reg_aux, _accu_fp16);
        }
    #pragma unroll
        for (uint32_t k = 0; k < 16; ++k){
            _accu[k]._vf.x = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[0]), _accu[k]._vf.x);
            _accu[k]._vf.y = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[1]), _accu[k]._vf.y);
            _accu[k]._vf.z = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[2]), _accu[k]._vf.z);
            _accu[k]._vf.w = __fadd_rn(__half2float(_accu_fp16[k]._v_half4[3]), _accu[k]._vf.w);
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
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            if (tidy * 16 + k < proc_dims_v1.y)  *((float4*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vf;
        }
    }
#endif
}
