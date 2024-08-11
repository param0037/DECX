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


#ifndef _GEMM_FP32_KERNEL_AARCH64_H_
#define _GEMM_FP32_KERNEL_AARCH64_H_

#include "../../../../../../common/basic.h"
#include "../../../../../../common/SIMD/intrinsics_ops.h"


namespace decx
{
namespace blas {
    namespace CPUK 
    {
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_fp32_dp_kernel_frag(const float* __restrict A_line, const float* __restrict B_lane,
            float* __restrict dst, const uint32_t _linear, const bool _first = false, const float* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_fp32_dp_kernel_frag_dual(const float* __restrict A_line, const float* __restrict B_lane,
            float* __restrict dst, const uint32_t _linear, const bool _first = false, const float* __restrict C = NULL);


        /*
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp32_block_kernel(const float* __restrict A, const float* __restrict B,
            float* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const float* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp32_kernel(const float* __restrict A, const float* __restrict B,
            float* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const float* __restrict C = NULL);
    }
}
}


template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp32_dp_kernel_frag(const float* __restrict    A_line, 
                         const float* __restrict    B_lane,
                         float* __restrict          dst, 
                         const uint32_t             _linear,
                         const bool                 _first,
                         const float* __restrict    C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm128_reg _accu;
    // first && !ABC: setzero
    // !first && !ABC: load dst
    // first && ABC: load C
    // !first && ABC: load dst
    if (!_first) { 
        _accu._vui = veorq_u32(_accu._vui, _accu._vui);
    }
    else {
        if constexpr (_ABC) { _accu._vf = vld1q_f32(C); }
        else { _accu._vui = veorq_u32(_accu._vui, _accu._vui); }
    }

    for (uint32_t i = 0; i < _linear / 4; ++i) 
    {
        float32x4_t A_palette = vld1q_f32(A_line + i * 4);

        float32x4_t B_v4 = vld1q_f32(B_lane + B_dex);
        // 0
        float32x4_t A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 0));
        _accu._vf = vfmaq_f32(_accu._vf, A_v4, B_v4);
        // 1
        A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 1));
        B_v4 = vld1q_f32(B_lane + B_dex + 8);
        _accu._vf = vfmaq_f32(_accu._vf, A_v4, B_v4);
        // 2
        A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 2));
        B_v4 = vld1q_f32(B_lane + B_dex + 16);
        _accu._vf = vfmaq_f32(_accu._vf, A_v4, B_v4);
        // 3
        A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 3));
        B_v4 = vld1q_f32(B_lane + B_dex + 24);
        _accu._vf = vfmaq_f32(_accu._vf, A_v4, B_v4);
        B_dex += 32;
    }

    const uint32_t _linear_L = _linear % 4;
    if (_linear_L) {
        for (uint32_t i = 0; i < _linear_L; ++i) {
            float32x4_t A_v4 = vdupq_n_f32(A_line[(_linear / 4) * 4 + i]);
            float32x4_t B_v4 = vld1q_f32(B_lane + B_dex);
            _accu._vf = vfmaq_f32(_accu._vf, A_v4, B_v4);
            B_dex += 8;
        }
    }
    vst1q_f32(dst, _accu._vf);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp32_dp_kernel_frag_dual(const float* __restrict A_line, 
                              const float* __restrict B_lane,
                              float* __restrict       dst, 
                              const uint32_t          _linear,
                              const bool              _first,
                              const float* __restrict C)
{
    uint32_t B_dex = 0;
    
    decx::utils::simd::xmm256_reg _accu;

    if (!_first) {
        _accu._vf = vld1q_f32_x2(dst);
    }
    else {
        if constexpr (_ABC) {
            _accu._vf = vld1q_f32_x2(C);
        }
        else { 
            _accu._vui.val[0] = veorq_u32(_accu._vui.val[0], _accu._vui.val[0]);
            _accu._vui.val[1] = veorq_u32(_accu._vui.val[1], _accu._vui.val[1]);
        }
    }

    for (uint32_t i = 0; i < _linear / 4; ++i) 
    {
        float32x4_t A_palette = vld1q_f32(A_line + i * 4);

        // 0
        float32x4x2_t B_v8 = vld1q_f32_x2(B_lane + B_dex);
        float32x4_t A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 0));
        _accu._vf.val[0] = vfmaq_f32(_accu._vf.val[0], A_v4, B_v8.val[0]);
        _accu._vf.val[1] = vfmaq_f32(_accu._vf.val[1], A_v4, B_v8.val[1]);
        // 1
        A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 1));
        B_v8 = vld1q_f32_x2(B_lane + B_dex + 8);
        _accu._vf.val[0] = vfmaq_f32(_accu._vf.val[0], A_v4, B_v8.val[0]);
        _accu._vf.val[1] = vfmaq_f32(_accu._vf.val[1], A_v4, B_v8.val[1]);
        // 2
        A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 2));
        B_v8 = vld1q_f32_x2(B_lane + B_dex + 16);
        _accu._vf.val[0] = vfmaq_f32(_accu._vf.val[0], A_v4, B_v8.val[0]);
        _accu._vf.val[1] = vfmaq_f32(_accu._vf.val[1], A_v4, B_v8.val[1]);
        // 3
        A_v4 = vdupq_n_f32(vgetq_lane_f32(A_palette, 3));
        B_v8 = vld1q_f32_x2(B_lane + B_dex + 24);
        _accu._vf.val[0] = vfmaq_f32(_accu._vf.val[0], A_v4, B_v8.val[0]);
        _accu._vf.val[1] = vfmaq_f32(_accu._vf.val[1], A_v4, B_v8.val[1]);

        B_dex += 32;
    }
    const uint32_t _linear_L = _linear % 4;
    if (_linear_L) {
        for (uint32_t i = 0; i < _linear_L; ++i) {
            float32x4_t A_v8 = vdupq_n_f32(A_line[(_linear / 4) * 4 + i]);
            float32x4x2_t B_v8 = vld1q_f32_x2(B_lane + B_dex);
            _accu._vf.val[0] = vfmaq_f32(_accu._vf.val[0], A_v8, B_v8.val[0]);
            _accu._vf.val[1] = vfmaq_f32(_accu._vf.val[1], A_v8, B_v8.val[1]);
            B_dex += 8;
        }
    }
    vst1q_f32_x2(dst, _accu._vf);
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp32_block_kernel(const float* __restrict    A,
                                         const float* __restrict    B, 
                                         float* __restrict          dst,
                                         const uint2                proc_dims_v4,
                                         const decx::utils::frag_manager* fmgrL,
                                         const uint32_t             pitchA_v1, 
                                         const uint32_t             Llen, 
                                         const uint32_t             pitchdst_v1,
                                         const float* __restrict    C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    
    for (uint32_t k = 0; k < fmgrL->frag_num; ++k) 
    {
        const uint32_t _L_frag = k == fmgrL->frag_num - 1 ? fmgrL->last_frag_len : fmgrL->frag_len;
        A_dex = fmgrL->frag_len * k;

        for (uint32_t i = 0; i < proc_dims_v4.y; ++i) {
            B_dex = fmgrL->frag_len * k * 8;
            dst_dex = i * pitchdst_v1;
            for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
                decx::blas::CPUK::GEMM_fp32_dp_kernel_frag_dual<_ABC>(A + A_dex, B + B_dex, dst + dst_dex, _L_frag,
                    k == 0, C);
                B_dex += Llen * 8;
                dst_dex += 8;
            }
            if (proc_dims_v4.x % 2) {
                decx::blas::CPUK::GEMM_fp32_dp_kernel_frag<_ABC>(A + A_dex, B + B_dex, dst + dst_dex, _L_frag, k == 0,
                    C);
            }
            A_dex += pitchA_v1;
        }
    }
}


// Typically the processing sizes = [BW, BH]
template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp32_kernel(const float* __restrict                  A, 
                                   const float* __restrict                  B,
                                   float* __restrict                        dst, 
                                   const decx::blas::GEMM_blocking_config*  config,
                                   const uint32_t                           pitchA_v1, 
                                   const uint32_t                           Llen, 
                                   const uint32_t                           pitchdst_v1,
                                   const float* __restrict                  C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;

    for (uint32_t i = 0; i < config->_fmgr_W.frag_num; ++i) 
    {
        B_dex = i * config->_fmgr_W.frag_len * Llen * 4;
        A_dex = 0;
        dst_dex = i * config->_fmgr_W.frag_len * 4;

        uint2 proc_dims = make_uint2(i < config->_fmgr_W.frag_num - 1 ? 
                                     config->_fmgr_W.frag_len : config->_fmgr_W.last_frag_len,
                                     config->_fmgr_H.frag_len);

        for (uint32_t j = 0; j < config->_fmgr_H.frag_num - 1; ++j) 
        {
            decx::blas::CPUK::GEMM_fp32_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                        proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);

            A_dex += config->_fmgr_H.frag_len * pitchA_v1;
            dst_dex += config->_fmgr_H.frag_len * pitchdst_v1;
        }

        proc_dims.y = config->_fmgr_H.last_frag_len;
        decx::blas::CPUK::GEMM_fp32_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                    proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);
    }
}


#endif