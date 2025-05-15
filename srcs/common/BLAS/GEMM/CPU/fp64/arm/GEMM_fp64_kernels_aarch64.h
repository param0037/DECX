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


#ifndef _GEMM_FP64_KERNEL_AARCH64_H_
#define _GEMM_FP64_KERNEL_AARCH64_H_

#include "../../../../../../common/basic.h"


namespace decx
{
namespace blas {
    namespace CPUK 
    {
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_fp64_dp_kernel_frag(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_fp64_dp_kernel_frag_dual(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);


        /**
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp64_block_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp64_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const double* __restrict C = NULL);
    }
}
}


template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp64_dp_kernel_frag(const double* __restrict    A_line, 
                         const double* __restrict    B_lane,
                         double* __restrict          dst, 
                         const uint32_t             _linear,
                         const bool                 _first,
                         const double* __restrict    C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm128_reg _accu;
    // first && !ABC: setzero
    // !first && !ABC: load dst
    // first && ABC: load C
    // !first && ABC: load dst
    if (!_first) { 
        _accu._vd = vld1q_f64(dst); 
    }
    else {
        if constexpr (_ABC) { _accu._vd = vld1q_f64(C); }
        else { _accu._vui = veorq_u32(_accu._vui, _accu._vui); }
    }

    for (uint32_t i = 0; i < _linear / 2; ++i) 
    {
        float64x2_t A_palette = vld1q_f64(A_line + i * 2);

        // 0
        float64x2_t A_v2 = vdupq_n_f64(vgetq_lane_f64(A_palette, 0));
        float64x2_t B_v2 = vld1q_f64(B_lane + B_dex);
        _accu._vd = vfmaq_f64(_accu._vd, A_v2, B_v2);
        // 1
        A_v2 = vdupq_n_f64(vgetq_lane_f64(A_palette, 1));
        B_v2 = vld1q_f64(B_lane + B_dex + 4);
        _accu._vd = vfmaq_f64(_accu._vd, A_v2, B_v2);
        B_dex += 8;
    }

    if (_linear & 1) {
        float64x2_t A_v2 = vdupq_n_f64(A_line[(_linear / 2) * 2]);
        float64x2_t B_v2 = vld1q_f64(B_lane + B_dex);
        _accu._vd = vfmaq_f64(_accu._vd, A_v2, B_v2);
    }
    vst1q_f64(dst, _accu._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp64_dp_kernel_frag_dual(const double* __restrict  A_line, 
                              const double* __restrict  B_lane,
                              double* __restrict        dst, 
                              const uint32_t            _linear,
                              const bool                _first,
                              const double* __restrict  C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu;
    if (!_first) {
        _accu._vd = vld1q_f64_x2(dst);
    }
    else {
        if constexpr (_ABC) {
            _accu._vd = vld1q_f64_x2(C);
        }
        else {
            _accu._vui.val[0] = veorq_u32(_accu._vui.val[0], _accu._vui.val[0]);
            _accu._vui.val[1] = veorq_u32(_accu._vui.val[1], _accu._vui.val[1]);
        }
    }

    for (uint32_t i = 0; i < _linear / 2; ++i) 
    {
        float64x2_t A_palette = vld1q_f64(A_line + i * 2);

        // 0
        float64x2_t A_v2 = vdupq_n_f64(vgetq_lane_f64(A_palette, 0));
        float64x2x2_t B_v4 = vld1q_f64_x2(B_lane + B_dex);
        _accu._vd.val[0] = vfmaq_f64(_accu._vd.val[0], A_v2, B_v4.val[0]);
        _accu._vd.val[1] = vfmaq_f64(_accu._vd.val[1], A_v2, B_v4.val[1]);
        // 1
        A_v2 = vdupq_n_f64(vgetq_lane_f64(A_palette, 1));
        B_v4 = vld1q_f64_x2(B_lane + B_dex + 4);
        _accu._vd.val[0] = vfmaq_f64(_accu._vd.val[0], A_v2, B_v4.val[0]);
        _accu._vd.val[1] = vfmaq_f64(_accu._vd.val[1], A_v2, B_v4.val[1]);
        B_dex += 8;
    }

    if (_linear & 1) {
        float64x2_t A_v2 = vdupq_n_f64(A_line[(_linear / 2) * 2]);
        float64x2x2_t B_v4 = vld1q_f64_x2(B_lane + B_dex);
        _accu._vd.val[0] = vfmaq_f64(_accu._vd.val[0], A_v2, B_v4.val[0]);
        _accu._vd.val[1] = vfmaq_f64(_accu._vd.val[1], A_v2, B_v4.val[1]);
    }

    vst1q_f64_x2(dst, _accu._vd);
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp64_block_kernel(const double* __restrict    A,
                                         const double* __restrict    B, 
                                         double* __restrict          dst,
                                         const uint2                proc_dims_v4,
                                         const decx::utils::frag_manager* fmgrL,
                                         const uint32_t             pitchA_v1, 
                                         const uint32_t             Llen, 
                                         const uint32_t             pitchdst_v1,
                                         const double* __restrict    C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    
    for (uint32_t k = 0; k < fmgrL->frag_num; ++k) 
    {
        const uint32_t _L_frag = k == fmgrL->frag_num - 1 ? fmgrL->last_frag_len : fmgrL->frag_len;
        A_dex = fmgrL->frag_len * k;

        for (uint32_t i = 0; i < proc_dims_v4.y; ++i) {
            B_dex = fmgrL->frag_len * k * 4;
            dst_dex = i * pitchdst_v1;
            for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
                decx::blas::CPUK::GEMM_fp64_dp_kernel_frag_dual<_ABC>(A + A_dex, B + B_dex, dst + dst_dex, _L_frag,
                    k == 0, C + dst_dex);
                B_dex += Llen * 4;
                dst_dex += 4;
            }
            if (proc_dims_v4.x % 2) {
                decx::blas::CPUK::GEMM_fp64_dp_kernel_frag<_ABC>(A + A_dex, B + B_dex, dst + dst_dex, _L_frag, k == 0,
                    C + dst_dex);
            }
            A_dex += pitchA_v1;
        }
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp64_kernel(const double* __restrict                  A, 
                                   const double* __restrict                  B,
                                   double* __restrict                        dst, 
                                   const decx::blas::GEMM_blocking_config*   config,
                                   const uint32_t                            pitchA_v1, 
                                   const uint32_t                            Llen, 
                                   const uint32_t                            pitchdst_v1,
                                   const double* __restrict                  C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;

    for (uint32_t i = 0; i < config->_fmgr_W.frag_num; ++i) 
    {
        B_dex = i * config->_fmgr_W.frag_len * Llen * 2;
        A_dex = 0;
        dst_dex = i * config->_fmgr_W.frag_len * 2;

        uint2 proc_dims = make_uint2(i < config->_fmgr_W.frag_num - 1 ? 
                                     config->_fmgr_W.frag_len : config->_fmgr_W.last_frag_len,
                                     config->_fmgr_H.frag_len);

        for (uint32_t j = 0; j < config->_fmgr_H.frag_num - 1; ++j) 
        {
            decx::blas::CPUK::GEMM_fp64_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                        proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);

            A_dex += config->_fmgr_H.frag_len * pitchA_v1;
            dst_dex += config->_fmgr_H.frag_len * pitchdst_v1;
        }

        proc_dims.y = config->_fmgr_H.last_frag_len;
        decx::blas::CPUK::GEMM_fp64_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                    proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);
    }
}


#endif
