/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_FP64_KERNELS_H_
#define _GEMM_FP64_KERNELS_H_

#include "../../../../core/basic.h"


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
            double* __restrict dst, const uint32_t _linear, const uint32_t pitchB_v8, const bool _first = false, const double* __restrict C = NULL);


        template <bool dual_lane, bool _ABC>
        static _THREAD_CALL_ void GEMM_fp64_dp_kernel(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const decx::utils::frag_manager* _fmgr_L, const uint32_t pitchB_v8 = 0, const double* __restrict C = NULL);

        /*
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp64_block_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t pitchB_v8, const uint32_t pitchdst_v1, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp64_kernel(const double* __restrict A, const double* __restrict B,
            double* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t pitchB_v8, const uint32_t pitchdst_v1, const double* __restrict C = NULL);

    }
}
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp64_dp_kernel_frag(const double* __restrict    A_line, 
                         const double* __restrict    B_lane,
                         double* __restrict          dst,
                         const uint32_t              _linear,
                         const bool                  _first,
                         const double* __restrict    C)
{
    uint32_t B_dex = 0;
    __m256d _accu;
    // first && !ABC: setzero
    // !first && !ABC: load dst
    // first && ABC: load C
    // !first && ABC: load dst
    if (!_first) {
        _accu = _mm256_load_pd(dst); 
    }
    else {
        if constexpr (_ABC) { _accu = _mm256_load_pd(C); }
        else { _accu = _mm256_setzero_pd(); }
    }

    for (uint32_t i = 0; i < _linear / 2; ++i) 
    {
        __m128d A_palette = _mm_load_pd(A_line + i * 2);
        __m256d A_palette2 = _mm256_castpd128_pd256(A_palette);
        A_palette2 = _mm256_insertf128_pd(A_palette2, A_palette, 1);

        // 0
        __m256d A_v8 = _mm256_permute_pd(A_palette2, 0b0000);
        __m256d B_v8 = _mm256_load_pd(B_lane + B_dex);
        _accu = _mm256_fmadd_pd(A_v8, B_v8, _accu);
        // 1
        A_v8 = _mm256_permute_pd(A_palette2, 0b1111);
        B_v8 = _mm256_load_pd(B_lane + B_dex + 4);
        _accu = _mm256_fmadd_pd(A_v8, B_v8, _accu);
        B_dex += 8;
    }

    if (_linear & 1) {
        __m256d A_v4 = _mm256_broadcast_sd(A_line + (_linear / 2) * 2);
        __m256d B_v4 = _mm256_load_pd(B_lane + B_dex);
        _accu = _mm256_fmadd_pd(A_v4, B_v4, _accu);
    }
    _mm256_store_pd(dst, _accu);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp64_dp_kernel_frag_dual(const double* __restrict A_line, 
                              const double* __restrict B_lane,
                              double* __restrict       dst, 
                              const uint32_t          _linear,
                              const uint32_t          pitchB_v4,
                              const bool              _first,
                              const double* __restrict C)
{
    uint32_t B_dex = 0;
    __m256d _accu[2];
    if (!_first) {
        _accu[0] = _mm256_load_pd(dst);
        _accu[1] = _mm256_load_pd(dst + 4);
    }
    else {
        if constexpr (_ABC) { 
            _accu[0] = _mm256_load_pd(C);
            _accu[1] = _mm256_load_pd(C + 4);
        }
        else { 
            _accu[0] = _mm256_setzero_pd();
            _accu[1] = _mm256_setzero_pd();
        }
    }

    for (uint32_t i = 0; i < _linear / 2; ++i) 
    {
        __m128d A_palette = _mm_load_pd(A_line + i * 2);
        __m256d A_palette2 = _mm256_castpd128_pd256(A_palette);
        A_palette2 = _mm256_insertf128_pd(A_palette2, A_palette, 1);

        // 0
        __m256d A_v8 = _mm256_permute_pd(A_palette2, 0b0000);
        __m256d B_v8_0 = _mm256_load_pd(B_lane + B_dex);
        _accu[0] = _mm256_fmadd_pd(A_v8, B_v8_0, _accu[0]);
        __m256d B_v4_1 = _mm256_load_pd(B_lane + B_dex + pitchB_v4 * 4);
        _accu[1] = _mm256_fmadd_pd(A_v8, B_v4_1, _accu[1]);
        // 1
        A_v8 = _mm256_permute_pd(A_palette2, 0b1111);
        B_v8_0 = _mm256_load_pd(B_lane + B_dex + 4);
        _accu[0] = _mm256_fmadd_pd(A_v8, B_v8_0, _accu[0]);
        B_v4_1 = _mm256_load_pd(B_lane + B_dex + 4 + pitchB_v4 * 4);
        _accu[1] = _mm256_fmadd_pd(A_v8, B_v4_1, _accu[1]);
        B_dex += 8;
    }

    if (_linear & 1) {
        __m256d A_v4 = _mm256_broadcast_sd(A_line + (_linear / 2) * 2);
        __m256d B_v4_0 = _mm256_load_pd(B_lane + B_dex);
        __m256d B_v4_1 = _mm256_load_pd(B_lane + B_dex + pitchB_v4 * 4);
        _accu[0] = _mm256_fmadd_pd(A_v4, B_v4_0, _accu[0]);
        _accu[1] = _mm256_fmadd_pd(A_v4, B_v4_1, _accu[1]);
    }
    _mm256_store_pd(dst, _accu[0]);
    _mm256_store_pd(dst + 4, _accu[1]);
}



template <bool dual_lane, bool _ABC>
static _THREAD_CALL_ void
decx::blas::CPUK::GEMM_fp64_dp_kernel(const double* __restrict A_line, 
                                      const double* __restrict B_lane,
                                      double* __restrict dst, 
                                      const decx::utils::frag_manager* _fmgr_L,
                                      const uint32_t pitchB_v4,
                                      const double* __restrict C)
{
    uint32_t A_dex = 0, B_dex = 0;

    for (uint32_t i = 0; i < _fmgr_L->frag_num - 1; ++i) {
        if constexpr (dual_lane) {
            decx::blas::CPUK::GEMM_fp64_dp_kernel_frag_dual<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->frag_len, 
                pitchB_v4, i == 0, C);
        }
        else {
            decx::blas::CPUK::GEMM_fp64_dp_kernel_frag<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->frag_len, i == 0,
                C);
        }
        A_dex += _fmgr_L->frag_len;
        B_dex += _fmgr_L->frag_len * 4;
    }
    if constexpr (dual_lane) {
        decx::blas::CPUK::GEMM_fp64_dp_kernel_frag_dual<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->last_frag_len,
            pitchB_v4, _fmgr_L->frag_num - 1 == 0, C);
    }
    else {
        decx::blas::CPUK::GEMM_fp64_dp_kernel_frag<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->last_frag_len,
            _fmgr_L->frag_num - 1 == 0, C);
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp64_block_kernel(const double* __restrict       A,
                                         const double* __restrict       B, 
                                         double* __restrict             dst,
                                         const uint2                    proc_dims_v4,
                                         const decx::utils::frag_manager* fmgrL,
                                         const uint32_t                 pitchA_v1, 
                                         const uint32_t                 pitchB_v4, 
                                         const uint32_t                 pitchdst_v1,
                                         const double* __restrict       C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    
    for (uint32_t i = 0; i < proc_dims_v4.y; ++i) {
        B_dex = 0;
        dst_dex = i * pitchdst_v1;
        for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
            decx::blas::CPUK::GEMM_fp64_dp_kernel<true, _ABC>(A + A_dex, B + B_dex, dst + dst_dex, fmgrL, pitchB_v4, C + dst_dex);
            B_dex += pitchB_v4 * 8;
            dst_dex += 8;
        }
        if (proc_dims_v4.x % 2) {
            decx::blas::CPUK::GEMM_fp64_dp_kernel<false, _ABC>(A + A_dex, B + B_dex, dst + dst_dex, fmgrL, 0, C + dst_dex);
        }
        A_dex += pitchA_v1;
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp64_kernel(const double* __restrict                  A, 
                                   const double* __restrict                  B,
                                   double* __restrict                        dst, 
                                   const decx::blas::GEMM_blocking_config*  config,
                                   const uint32_t                           pitchA_v1, 
                                   const uint32_t                           pitchB_v4, 
                                   const uint32_t                           pitchdst_v1,
                                   const double* __restrict                  C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;

    for (uint32_t i = 0; i < config->_fmgr_H.frag_num; ++i) 
    {
        B_dex = 0;      // To the origin when a new row begin
        dst_dex = i * config->_fmgr_H.frag_len * pitchdst_v1;       // Targeted to the next row

        uint2 proc_dims = make_uint2(config->_fmgr_W.frag_len,
                                     i < config->_fmgr_H.frag_num - 1 ? 
                                         config->_fmgr_H.frag_len : 
                                         config->_fmgr_H.last_frag_len);

        for (uint32_t j = 0; j < config->_fmgr_W.frag_num - 1; ++j) 
        {
            decx::blas::CPUK::GEMM_fp64_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                proc_dims, &config->_fmgr_L, pitchA_v1, pitchB_v4, pitchdst_v1, C + dst_dex);

            B_dex += config->_fmgr_W.frag_len * pitchB_v4 * 4;
            dst_dex += config->_fmgr_W.frag_len * 4;
        }

        proc_dims.x = config->_fmgr_W.last_frag_len;
        decx::blas::CPUK::GEMM_fp64_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
            proc_dims, &config->_fmgr_L, pitchA_v1, pitchB_v4, pitchdst_v1, C + dst_dex);

        A_dex += config->_fmgr_H.frag_len * pitchA_v1;
    }
}


#endif
