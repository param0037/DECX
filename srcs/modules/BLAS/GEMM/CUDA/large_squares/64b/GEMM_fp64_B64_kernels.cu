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


#define _DECL_A_FP64_DEX_OUTER_(_row_id) (_row_id >> 1)
#define _DECL_A_FP64_DEX_INNER_(_row_id) (_row_id & 1)
#define _DECL_A_ARG_FP64_(regsA, _row_id) (regsA[_DECL_A_FP64_DEX_OUTER_(_row_id)]._arrd[_DECL_A_FP64_DEX_INNER_(_row_id)])


#define _MMA_FP64_1_2_1_(regsA, regsB, accu, _row_id) {     \
    accu[_row_id]._vd.x = __fma_rn(_DECL_A_ARG_FP64_(regsA, _row_id), regsB._vd.x, accu[_row_id]._vd.x);    \
    accu[_row_id]._vd.y = __fma_rn(_DECL_A_ARG_FP64_(regsA, _row_id), regsB._vd.y, accu[_row_id]._vd.y);    \
}


#define _MMA_FP64_1_2_8_(regsA, regsB, accu) {      \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 0);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 1);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 2);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 3);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 4);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 5);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 6);        \
    _MMA_FP64_1_2_1_(regsA, regsB, accu, 7);        \
}


__global__ void decx::blas::GPUK::
cu_GEMM_fp64_kernel_16_64_64(const double* __restrict A,   const double* __restrict B, 
                            double* __restrict dst,       const uint2 proc_dims_v1, 
                            const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                            const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 16 / 1;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 16 / 8;
    constexpr uint32_t _LDG_HA_step = 64 / _loc_LDG_Ay;
    
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay + blockIdx.y * 64;

    const uint32_t W_v2 = decx::utils::fast_uint_ceil2<uint32_t>(proc_dims_v1.x);

    __shared__ double _frag_A[64][16 + 1];
    __shared__ double2 _frag_B[16][64 / 2];

    decx::utils::_cuda_vec128 _accu[8];
    decx::utils::_cuda_vec128 regs[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx * 2 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 16); ++i)
    {
        // Load from A
        if (_Lloc_A < _L_v1){
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                regs[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k * 16 < proc_dims_v1.y) regs[k]._vd.x = A[dex_A + k * pitchA_v1 * 16];
                _frag_A[loc_tid_Ay + k * 16][loc_tid_Ax] = regs[k]._vd.x;
            }
        }
        // Load from B
        if (tidx < W_v2){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                regs[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) regs[k]._vd = *((double2*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = regs[k]._vd;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 16; ++_l)
        {
            regs[0]._vd.x = _frag_A[threadIdx.y * 8][_l];
            regs[0]._vd.y = _frag_A[threadIdx.y * 8 + 1][_l];
            regs[1]._vd.x = _frag_A[threadIdx.y * 8 + 2][_l];
            regs[1]._vd.y = _frag_A[threadIdx.y * 8 + 3][_l];
            regs[2]._vd.x = _frag_A[threadIdx.y * 8 + 4][_l];
            regs[2]._vd.y = _frag_A[threadIdx.y * 8 + 5][_l];
            regs[3]._vd.x = _frag_A[threadIdx.y * 8 + 6][_l];
            regs[3]._vd.y = _frag_A[threadIdx.y * 8 + 7][_l];

            _reg_aux._vd = _frag_B[_l][threadIdx.x];

            _MMA_FP64_1_2_8_(regs, _reg_aux, _accu);
        }

        _Lloc_A += 16;
        _Lloc_B += 16;

        dex_A += 16;
        dex_B += 16 * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 2 + tidy * pitchdst_v1 * 8;

    if (tidx < decx::utils::fast_uint_ceil2<uint32_t>(proc_dims_v1.x))
    {
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            if (tidy + k < proc_dims_v1.y)  *((double2*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vd;
        }
    }
}



__global__ void decx::blas::GPUK::
cu_GEMM_fp64_F_kernel_16_64_64(const double* __restrict A,   const double* __restrict B, 
                               const double* __restrict C,   double* __restrict dst,       
                               const double alpha,           const double beta,
                               const uint2 proc_dims_v1,     const uint32_t _L_v1,        
                               const uint32_t pitchA_v1,     const uint32_t pitchB_v1,    
                               const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 16 / 1;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 16 / 8;
    constexpr uint32_t _LDG_HA_step = 64 / _loc_LDG_Ay;
    
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay + blockIdx.y * 64;

    const uint32_t W_v2 = decx::utils::fast_uint_ceil2<uint32_t>(proc_dims_v1.x);

    __shared__ double _frag_A[64][16 + 1];
    __shared__ double2 _frag_B[16][64 / 2];

    decx::utils::_cuda_vec128 _accu[8];
    decx::utils::_cuda_vec128 regs[4], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx * 2 + pitchB_v1 * _Lloc_B;

    // Initialize the accumulators to all zeros.
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k){
        _accu[k]._vf = decx::utils::vec4_set1_fp32(0);
    }

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(_L_v1, 16); ++i)
    {
        // Load from A
        if (_Lloc_A < _L_v1){
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                regs[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (tid_Ay + k * 16 < proc_dims_v1.y) regs[k]._vd.x = A[dex_A + k * pitchA_v1 * 16];
                _frag_A[loc_tid_Ay + k * 16][loc_tid_Ax] = regs[k]._vd.x;
            }
        }
        // Load from B
        if (tidx < W_v2){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                regs[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) regs[k]._vd = *((double2*)(B + dex_B + k * pitchB_v1));
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = regs[k]._vd;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 16; ++_l)
        {
            regs[0]._vd.x = _frag_A[threadIdx.y * 8][_l];
            regs[0]._vd.y = _frag_A[threadIdx.y * 8 + 1][_l];
            regs[1]._vd.x = _frag_A[threadIdx.y * 8 + 2][_l];
            regs[1]._vd.y = _frag_A[threadIdx.y * 8 + 3][_l];
            regs[2]._vd.x = _frag_A[threadIdx.y * 8 + 4][_l];
            regs[2]._vd.y = _frag_A[threadIdx.y * 8 + 5][_l];
            regs[3]._vd.x = _frag_A[threadIdx.y * 8 + 6][_l];
            regs[3]._vd.y = _frag_A[threadIdx.y * 8 + 7][_l];

            _reg_aux._vd = _frag_B[_l][threadIdx.x];

            _MMA_FP64_1_2_8_(regs, _reg_aux, _accu);
        }

        _Lloc_A += 16;
        _Lloc_B += 16;

        dex_A += 16;
        dex_B += 16 * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx * 2 + tidy * pitchdst_v1 * 8;

    if (tidx < decx::utils::fast_uint_ceil2<uint32_t>(proc_dims_v1.x))
    {
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) 
        {
            _accu[k]._vd.x = __dmul_rn(_accu[k]._vd.x, alpha);
            _accu[k]._vd.y = __dmul_rn(_accu[k]._vd.y, alpha);

            if (tidy * 8 + k < proc_dims_v1.y)  _reg_aux._vd = *((double2*)(C + dex_dst + k * pitchB_v1));

            _accu[k]._vd.x = __fma_rn(_reg_aux._vd.x, beta, _accu[k]._vd.x);
            _accu[k]._vd.y = __fma_rn(_reg_aux._vd.y, beta, _accu[k]._vd.y);

            if (tidy * 8 + k < proc_dims_v1.y)  *((double2*)(dst + dex_dst + k * pitchdst_v1)) = _accu[k]._vd;
            //if (tidy * 8 + k < proc_dims_v1.y)  *((double2*)(dst + dex_dst + k * pitchdst_v1)) = decx::utils::vec2_set1_fp64(37);
        }
    }
}