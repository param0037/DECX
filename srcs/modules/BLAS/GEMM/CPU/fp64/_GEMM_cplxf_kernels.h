/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef __GEMM_CPLXF_KERNEL_H_
#define __GEMM_CPLXF_KERNEL_H_

#include "../../../../core/basic.h"
#include "../../../../DSP/CPU_cpf32_avx.h"


namespace decx
{
namespace blas {
    namespace CPUK 
    {
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_frag(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_frag_dual(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);


        template <bool dual_lane, bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const decx::utils::frag_manager* _fmgr_L, const double* __restrict C = NULL);


        /**
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_cplxf_block_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_cplxf_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const double* __restrict C = NULL);
    }
}
}


template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_frag(const double* __restrict    A_line, 
                          const double* __restrict    B_lane,
                          double* __restrict          dst, 
                          const uint32_t              _linear,
                          const bool                  _first,
                          const double* __restrict    C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu;
    // first && !ABC: setzero
    // !first && !ABC: load dst
    // first && ABC: load C
    // !first && ABC: load dst
    if (!_first) { 
        _accu._vd = _mm256_load_pd(dst); 
    }
    else {
        if constexpr (_ABC) { _accu._vd = _mm256_load_pd(C); }
        else { _accu._vd = _mm256_setzero_pd(); }
    }

    for (uint32_t i = 0; i < _linear / 2; ++i) 
    {
        __m128d A_palette = _mm_load_pd(A_line + i * 2);
        __m256d A_palette2 = _mm256_castpd128_pd256(A_palette);
        A_palette2 = _mm256_insertf128_pd(A_palette2, A_palette, 1);

        // 0
        decx::utils::simd::xmm256_reg A_v8;
        A_v8._vd = _mm256_permute_pd(A_palette2, 0b0000);
        decx::utils::simd::xmm256_reg B_v8;
        B_v8._vd = _mm256_load_pd(B_lane + B_dex);
        _accu._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v8._vf, B_v8._vf, _accu._vf);
        // 1
        A_v8._vd = _mm256_permute_pd(A_palette2, 0b1111);
        B_v8._vd = _mm256_load_pd(B_lane + B_dex + 8);
        _accu._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v8._vf, B_v8._vf, _accu._vf);
        B_dex += 16;
    }

    if (_linear & 1) {
        decx::utils::simd::xmm256_reg A_v4;
        A_v4._vd = _mm256_broadcast_sd(A_line + (_linear / 2) * 2);
        decx::utils::simd::xmm256_reg B_v4;
        B_v4._vd = _mm256_load_pd(B_lane + B_dex);
        _accu._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v4._vf, B_v4._vf, _accu._vf);
    }
    _mm256_store_pd(dst, _accu._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_frag_dual(const double* __restrict  A_line, 
                               const double* __restrict  B_lane,
                               double* __restrict        dst, 
                               const uint32_t            _linear,
                               const bool                _first,
                               const double* __restrict  C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu[2];

    if (!_first) {
        _accu[0]._vd = _mm256_load_pd(dst);
        _accu[1]._vd = _mm256_load_pd(dst + 4);
    }
    else {
        if constexpr (_ABC) {
            _accu[0]._vd = _mm256_load_pd(C);
            _accu[1]._vd = _mm256_load_pd(C + 4);
        }
        else {
            _accu[0]._vd = _mm256_setzero_pd();
            _accu[1]._vd = _mm256_setzero_pd();
        }
    }

    for (uint32_t i = 0; i < _linear / 2; ++i) 
    {
        __m128d A_palette = _mm_load_pd(A_line + i * 2);
        __m256d A_palette2 = _mm256_castpd128_pd256(A_palette);
        A_palette2 = _mm256_insertf128_pd(A_palette2, A_palette, 1);

        // 0
        decx::utils::simd::xmm256_reg A_v8;
        A_v8._vd = _mm256_permute_pd(A_palette2, 0b0000);
        decx::utils::simd::xmm256_reg B_v8_0;
        B_v8_0._vd = _mm256_load_pd(B_lane + B_dex);
        _accu[0]._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v8._vf, B_v8_0._vf, _accu[0]._vf);
        decx::utils::simd::xmm256_reg B_v4_1;
        B_v4_1._vd = _mm256_load_pd(B_lane + B_dex + 4);
        _accu[1]._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v8._vf, B_v4_1._vf, _accu[1]._vf);
        // 1
        A_v8._vd = _mm256_permute_pd(A_palette2, 0b1111);
        B_v8_0._vd = _mm256_load_pd(B_lane + B_dex + 8);
        _accu[0]._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v8._vf, B_v8_0._vf, _accu[0]._vf);
        B_v4_1._vd = _mm256_load_pd(B_lane + B_dex + 12);
        _accu[1]._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v8._vf, B_v4_1._vf, _accu[1]._vf);
        B_dex += 16;
    }

    if (_linear & 1) {
        decx::utils::simd::xmm256_reg A_v4;
        A_v4._vd = _mm256_broadcast_sd(A_line + (_linear / 2) * 2);
        decx::utils::simd::xmm256_reg B_v4_0;
        B_v4_0._vd = _mm256_load_pd(B_lane + B_dex);
        decx::utils::simd::xmm256_reg B_v4_1;
        B_v4_1._vd = _mm256_load_pd(B_lane + B_dex + 4);
        _accu[0]._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v4._vf, B_v4_0._vf, _accu[0]._vf);
        _accu[1]._vf = decx::dsp::CPUK::_cp4_fma_cp4_fp32(A_v4._vf, B_v4_1._vf, _accu[1]._vf);
    }
    _mm256_store_pd(dst, _accu[0]._vd);
    _mm256_store_pd(dst + 4, _accu[1]._vd);
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxf_block_kernel(const double* __restrict    A,
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
            B_dex = fmgrL->frag_len * k * 8;
            dst_dex = i * pitchdst_v1;
            for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_frag_dual<_ABC>(A + A_dex, B + B_dex, dst + dst_dex, _L_frag,
                    k == 0, C);
                B_dex += Llen * 8;
                dst_dex += 8;
            }
            if (proc_dims_v4.x % 2) {
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_frag<_ABC>(A + A_dex, B + B_dex, dst + dst_dex, _L_frag, k == 0,
                    C);
            }
            A_dex += pitchA_v1;
        }
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxf_kernel(const double* __restrict                  A, 
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
        B_dex = i * config->_fmgr_W.frag_len * Llen * 4;
        A_dex = 0;
        dst_dex = i * config->_fmgr_W.frag_len * 4;

        uint2 proc_dims = make_uint2(i < config->_fmgr_W.frag_num - 1 ? 
                                     config->_fmgr_W.frag_len : config->_fmgr_W.last_frag_len,
                                     config->_fmgr_H.frag_len);

        for (uint32_t j = 0; j < config->_fmgr_H.frag_num - 1; ++j) 
        {
            decx::blas::CPUK::GEMM_cplxf_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                        proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);

            A_dex += config->_fmgr_H.frag_len * pitchA_v1;
            dst_dex += config->_fmgr_H.frag_len * pitchdst_v1;
        }

        proc_dims.y = config->_fmgr_H.last_frag_len;
        decx::blas::CPUK::GEMM_cplxf_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                    proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);
    }
}

#endif
