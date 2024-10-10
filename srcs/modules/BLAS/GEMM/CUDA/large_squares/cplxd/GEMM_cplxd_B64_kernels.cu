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
#include "../../../../../../common/CUSV/CUDA_cpd64.cuh"


#define _CPLXD_FMA_(a, b, c, res) {     \
    res.real = __dsub_rn(__fma_rn(a.real, b.real, c.real), __dmul_rn(a.image, b.image));    \
    res.image = __dadd_rn(__fma_rn(a.real, b.image, c.image), __dmul_rn(a.image, b.real));  \
}


#define _CPLXD_MUL_(a, b, res) {     \
    res.real = __dsub_rn(__dmul_rn(a.real, b.real), __dmul_rn(a.image, b.image));   \
    res.image = __dadd_rn(__dmul_rn(a.real, b.image), __dmul_rn(a.image, b.real));  \
}


#define _MMA_CPLXD_1_1_8_(regsA, regsB, accu) {      \
    _CPLXD_FMA_(regsA[0]._cplxd, regsB._cplxd, accu[0]._cplxd, accu[0]._cplxd);        \
    _CPLXD_FMA_(regsA[1]._cplxd, regsB._cplxd, accu[1]._cplxd, accu[1]._cplxd);        \
    _CPLXD_FMA_(regsA[2]._cplxd, regsB._cplxd, accu[2]._cplxd, accu[2]._cplxd);        \
    _CPLXD_FMA_(regsA[3]._cplxd, regsB._cplxd, accu[3]._cplxd, accu[3]._cplxd);        \
    _CPLXD_FMA_(regsA[4]._cplxd, regsB._cplxd, accu[4]._cplxd, accu[4]._cplxd);        \
    _CPLXD_FMA_(regsA[5]._cplxd, regsB._cplxd, accu[5]._cplxd, accu[5]._cplxd);        \
    _CPLXD_FMA_(regsA[6]._cplxd, regsB._cplxd, accu[6]._cplxd, accu[6]._cplxd);        \
    _CPLXD_FMA_(regsA[7]._cplxd, regsB._cplxd, accu[7]._cplxd, accu[7]._cplxd);        \
}


__global__ void decx::blas::GPUK::
cu_GEMM_cplxd_kernel_16_32_64(const double2* __restrict A,   const double2* __restrict B, 
                            double2* __restrict dst,       const uint2 proc_dims_v1, 
                            const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                            const uint32_t pitchB_v1,    const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 16 / 1;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 16 / 8;
    
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay + blockIdx.y * 64;

    __shared__ double2 _frag_A[64][16 + 1];
    __shared__ double2 _frag_B[16][64];

    decx::utils::_cuda_vec128 _accu[8];
    decx::utils::_cuda_vec128 regs[8], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx + pitchB_v1 * _Lloc_B;

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
                if (tid_Ay + k * 16 < proc_dims_v1.y) regs[k]._vd = A[dex_A + k * pitchA_v1 * 16];
                _frag_A[loc_tid_Ay + k * 16][loc_tid_Ax] = regs[k]._vd;
            }
        }
        // Load from B
        if (tidx < proc_dims_v1.x){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                regs[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) regs[k]._vd = B[dex_B + k * pitchB_v1];
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = regs[k]._vd;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 16; ++_l)
        {
            regs[0]._vd = _frag_A[threadIdx.y * 8][_l];
            regs[1]._vd = _frag_A[threadIdx.y * 8 + 1][_l];
            regs[2]._vd = _frag_A[threadIdx.y * 8 + 2][_l];
            regs[3]._vd = _frag_A[threadIdx.y * 8 + 3][_l];
            regs[4]._vd = _frag_A[threadIdx.y * 8 + 4][_l];
            regs[5]._vd = _frag_A[threadIdx.y * 8 + 5][_l];
            regs[6]._vd = _frag_A[threadIdx.y * 8 + 6][_l];
            regs[7]._vd = _frag_A[threadIdx.y * 8 + 7][_l];

            _reg_aux._vd = _frag_B[_l][threadIdx.x];

            _MMA_CPLXD_1_1_8_(regs, _reg_aux, _accu);
        }

        _Lloc_A += 16;
        _Lloc_B += 16;

        dex_A += 16;
        dex_B += 16 * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx + tidy * pitchdst_v1 * 8;

    if (tidx < proc_dims_v1.x)
    {
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            if (tidy * 8 + k < proc_dims_v1.y)  dst[dex_dst + k * pitchdst_v1] = _accu[k]._vd;
        }
    }
}



__global__ void decx::blas::GPUK::
cu_GEMM_cplxd_F_kernel_16_32_64(const double2* __restrict A,    const double2* __restrict B, 
                                const double2* __restrict C,    double2* __restrict dst,       
                                const de::CPd alpha,            const de::CPd beta, 
                                const uint2 proc_dims_v1,       const uint32_t _L_v1,        
                                const uint32_t pitchA_v1,       const uint32_t pitchB_v1,
                                const uint32_t pitchdst_v1)
{
    constexpr uint32_t _loc_LDG_Ax = 16 / 1;
    constexpr uint32_t _loc_LDG_Ay = 256 / _loc_LDG_Ax;
    constexpr uint32_t _LDG_HB_step = 16 / 8;
    
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t loc_tid_1d = threadIdx.x + threadIdx.y * blockDim.x;

    // Rearrange the 2D thread layout from 8x32 to 32x8 for LDG from A
    const uint32_t loc_tid_Ax = loc_tid_1d % _loc_LDG_Ax;
    const uint32_t loc_tid_Ay = loc_tid_1d / _loc_LDG_Ax;
    const uint32_t tid_Ay = loc_tid_Ay + blockIdx.y * 64;

    __shared__ double2 _frag_A[64][16 + 1];
    __shared__ double2 _frag_B[16][64];

    decx::utils::_cuda_vec128 _accu[8];
    decx::utils::_cuda_vec128 regs[8], _reg_aux;

    uint32_t _Lloc_A = loc_tid_Ax;
    uint32_t _Lloc_B = threadIdx.y * _LDG_HB_step;
    
    uint64_t dex_A = _Lloc_A + pitchA_v1 * tid_Ay;
    uint64_t dex_B = tidx + pitchB_v1 * _Lloc_B;

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
                if (tid_Ay + k * 16 < proc_dims_v1.y) regs[k]._vd = A[dex_A + k * pitchA_v1 * 16];
                _frag_A[loc_tid_Ay + k * 16][loc_tid_Ax] = regs[k]._vd;
            }
        }
        // Load from B
        if (tidx < proc_dims_v1.x){
#pragma unroll
            for (uint32_t k = 0; k < _LDG_HB_step; ++k) {
                regs[k]._vf = decx::utils::vec4_set1_fp32(0);
                if (_Lloc_B + k < _L_v1) regs[k]._vd = B[dex_B + k * pitchB_v1];
                _frag_B[threadIdx.y * _LDG_HB_step + k][threadIdx.x] = regs[k]._vd;
            }
        }

        __syncthreads();

#pragma unroll
        for (uint32_t _l = 0; _l < 16; ++_l)
        {
            regs[0]._vd = _frag_A[threadIdx.y * 8][_l];
            regs[1]._vd = _frag_A[threadIdx.y * 8 + 1][_l];
            regs[2]._vd = _frag_A[threadIdx.y * 8 + 2][_l];
            regs[3]._vd = _frag_A[threadIdx.y * 8 + 3][_l];
            regs[4]._vd = _frag_A[threadIdx.y * 8 + 4][_l];
            regs[5]._vd = _frag_A[threadIdx.y * 8 + 5][_l];
            regs[6]._vd = _frag_A[threadIdx.y * 8 + 6][_l];
            regs[7]._vd = _frag_A[threadIdx.y * 8 + 7][_l];

            _reg_aux._vd = _frag_B[_l][threadIdx.x];

            _MMA_CPLXD_1_1_8_(regs, _reg_aux, _accu);
        }

        _Lloc_A += 16;
        _Lloc_B += 16;

        dex_A += 16;
        dex_B += 16 * pitchB_v1;

        __syncthreads();
    }

    // Store the results to dst.
    const uint64_t dex_dst = tidx + tidy * pitchdst_v1 * 8;

    if (tidx < proc_dims_v1.x)
    {
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) 
        {
            _CPLXD_MUL_(_accu[k]._cplxd, alpha, _accu[k]._cplxd);

            if (tidy * 8 + k < proc_dims_v1.y)  _reg_aux._vd = dst[dex_dst + k * pitchdst_v1];
            _CPLXD_FMA_(_reg_aux._cplxd, beta, _accu[k]._cplxd, _accu[k]._cplxd);

            if (tidy * 8 + k < proc_dims_v1.y)  dst[dex_dst + k * pitchdst_v1] = _accu[k]._vd;
        }
    }
}