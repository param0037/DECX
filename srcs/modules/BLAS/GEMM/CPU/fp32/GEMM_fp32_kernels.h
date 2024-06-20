/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_FP32_KERNEL_H_
#define _GEMM_FP32_KERNEL_H_

#include "../../../../core/basic.h"


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
            float* __restrict dst, const uint32_t _linear, const uint32_t pitchB_v8, const bool _first = false, const float* __restrict C = NULL);


        template <bool dual_lane, bool _ABC>
        static _THREAD_CALL_ void GEMM_fp32_dp_kernel(const float* __restrict A_line, const float* __restrict B_lane,
            float* __restrict dst, const decx::utils::frag_manager* _fmgr_L, const uint32_t pitchB_v8 = 0, const float* __restrict C = NULL);


        /*
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp32_block_kernel(const float* __restrict A, const float* __restrict B,
            float* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t pitchB_v8, const uint32_t pitchdst_v1, const float* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_fp32_kernel(const float* __restrict A, const float* __restrict B,
            float* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t pitchB_v8, const uint32_t pitchdst_v1, const float* __restrict C = NULL);
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
    __m256 _accu;
    // first && !ABC: setzero
    // !first && !ABC: load dst
    // first && ABC: load C
    // !first && ABC: load dst
    if (!_first) { 
        _accu = _mm256_load_ps(dst); 
    }
    else {
        if constexpr (_ABC) { _accu = _mm256_load_ps(C); }
        else { _accu = _mm256_setzero_ps(); }
    }

    for (uint32_t i = 0; i < _linear / 4; ++i) 
    {
        __m128 A_palette = _mm_load_ps(A_line + i * 4);
        __m256 A_palette2 = _mm256_castps128_ps256(A_palette);
        A_palette2 = _mm256_insertf128_ps(A_palette2, A_palette, 1);

        // 0
        __m256 A_v8 = _mm256_permute_ps(A_palette2, 0b00000000);
        __m256 B_v8 = _mm256_load_ps(B_lane + B_dex);
        _accu = _mm256_fmadd_ps(A_v8, B_v8, _accu);
        // 1
        A_v8 = _mm256_permute_ps(A_palette2, 0b01010101);
        B_v8 = _mm256_load_ps(B_lane + B_dex + 8);
        _accu = _mm256_fmadd_ps(A_v8, B_v8, _accu);
        // 2
        A_v8 = _mm256_permute_ps(A_palette2, 0b10101010);
        B_v8 = _mm256_load_ps(B_lane + B_dex + 16);
        _accu = _mm256_fmadd_ps(A_v8, B_v8, _accu);
        // 3
        A_v8 = _mm256_permute_ps(A_palette2, 0b11111111);
        B_v8 = _mm256_load_ps(B_lane + B_dex + 24);
        _accu = _mm256_fmadd_ps(A_v8, B_v8, _accu);
        B_dex += 32;
    }
    const uint32_t _linear_L = _linear % 4;
    if (_linear_L) {
        for (uint32_t i = 0; i < _linear_L; ++i) {
            __m256 A_v8 = _mm256_broadcast_ss(A_line + (_linear / 4) * 4 + i);
            __m256 B_v8 = _mm256_load_ps(B_lane + B_dex);
            _accu = _mm256_fmadd_ps(A_v8, B_v8, _accu);
            B_dex += 8;
        }
    }
    _mm256_store_ps(dst, _accu);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_fp32_dp_kernel_frag_dual(const float* __restrict A_line, 
                              const float* __restrict B_lane,
                              float* __restrict       dst, 
                              const uint32_t          _linear,
                              const uint32_t          pitchB_v8,
                              const bool              _first,
                              const float* __restrict C)
{
    uint32_t B_dex = 0;
    __m256 _accu[2];
    if (!_first) {
        _accu[0] = _mm256_load_ps(dst);
        _accu[1] = _mm256_load_ps(dst + 8);
    }
    else {
        if constexpr (_ABC) { 
            _accu[0] = _mm256_load_ps(C);
            _accu[1] = _mm256_load_ps(C + 8);
        }
        else { 
            _accu[0] = _mm256_setzero_ps();
            _accu[1] = _mm256_setzero_ps();
        }
    }

    for (uint32_t i = 0; i < _linear / 4; ++i) 
    {
        __m128 A_palette = _mm_load_ps(A_line + i * 4);
        __m256 A_palette2 = _mm256_castps128_ps256(A_palette);
        A_palette2 = _mm256_insertf128_ps(A_palette2, A_palette, 1);

        // 0
        __m256 A_v8 = _mm256_permute_ps(A_palette2, 0b00000000);
        __m256 B_v8_0 = _mm256_load_ps(B_lane + B_dex);
        _accu[0] = _mm256_fmadd_ps(A_v8, B_v8_0, _accu[0]);
        __m256 B_v8_1 = _mm256_load_ps(B_lane + B_dex + pitchB_v8 * 8);
        _accu[1] = _mm256_fmadd_ps(A_v8, B_v8_1, _accu[1]);
        // 1
        A_v8 = _mm256_permute_ps(A_palette2, 0b01010101);
        B_v8_0 = _mm256_load_ps(B_lane + B_dex + 8);
        _accu[0] = _mm256_fmadd_ps(A_v8, B_v8_0, _accu[0]);
        B_v8_1 = _mm256_load_ps(B_lane + B_dex + 8 + pitchB_v8 * 8);
        _accu[1] = _mm256_fmadd_ps(A_v8, B_v8_1, _accu[1]);
        // 2
        A_v8 = _mm256_permute_ps(A_palette2, 0b10101010);
        B_v8_0 = _mm256_load_ps(B_lane + B_dex + 16);
        _accu[0] = _mm256_fmadd_ps(A_v8, B_v8_0, _accu[0]);
        B_v8_1 = _mm256_load_ps(B_lane + B_dex + 16 + pitchB_v8 * 8);
        _accu[1] = _mm256_fmadd_ps(A_v8, B_v8_1, _accu[1]);
        // 3
        A_v8 = _mm256_permute_ps(A_palette2, 0b11111111);
        B_v8_0 = _mm256_load_ps(B_lane + B_dex + 24);
        _accu[0] = _mm256_fmadd_ps(A_v8, B_v8_0, _accu[0]);
        B_v8_1 = _mm256_load_ps(B_lane + B_dex + 24 + pitchB_v8 * 8);
        _accu[1] = _mm256_fmadd_ps(A_v8, B_v8_1, _accu[1]);
        B_dex += 32;
    }
    const uint32_t _linear_L = _linear % 4;
    if (_linear_L) {
        for (uint32_t i = 0; i < _linear_L; ++i) {
            __m256 A_v8 = _mm256_broadcast_ss(A_line + (_linear / 4) * 4 + i);
            __m256 B_v8_0 = _mm256_load_ps(B_lane + B_dex);
            __m256 B_v8_1 = _mm256_load_ps(B_lane + B_dex + pitchB_v8 * 8);
            _accu[0] = _mm256_fmadd_ps(A_v8, B_v8_0, _accu[0]);
            _accu[1] = _mm256_fmadd_ps(A_v8, B_v8_1, _accu[1]);
            B_dex += 8;
        }
    }
    _mm256_store_ps(dst, _accu[0]);
    _mm256_store_ps(dst + 8, _accu[1]);
}



template <bool dual_lane, bool _ABC>
static _THREAD_CALL_ void
decx::blas::CPUK::GEMM_fp32_dp_kernel(const float* __restrict A_line, 
                                      const float* __restrict B_lane,
                                      float* __restrict dst, 
                                      const decx::utils::frag_manager* _fmgr_L,
                                      const uint32_t pitchB_v8,
                                      const float* __restrict C)
{
    uint32_t A_dex = 0, B_dex = 0;

    for (uint32_t i = 0; i < _fmgr_L->frag_num - 1; ++i) {
        if constexpr (dual_lane) {
            decx::blas::CPUK::GEMM_fp32_dp_kernel_frag_dual<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->frag_len, 
                pitchB_v8, i == 0, C);
        }
        else {
            decx::blas::CPUK::GEMM_fp32_dp_kernel_frag<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->frag_len, i == 0,
                C);
        }
        A_dex += _fmgr_L->frag_len;
        B_dex += _fmgr_L->frag_len * 8;
    }
    if constexpr (dual_lane) {
        decx::blas::CPUK::GEMM_fp32_dp_kernel_frag_dual<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->last_frag_len,
            pitchB_v8, _fmgr_L->frag_num - 1 == 0, C);
    }
    else {
        decx::blas::CPUK::GEMM_fp32_dp_kernel_frag<_ABC>(A_line + A_dex, B_lane + B_dex, dst, _fmgr_L->last_frag_len,
            _fmgr_L->frag_num - 1 == 0, C);
    }
}


template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp32_block_kernel(const float* __restrict    A,
                                         const float* __restrict    B, 
                                         float* __restrict          dst,
                                         const uint2                proc_dims_v8,
                                         const decx::utils::frag_manager* fmgrL,
                                         const uint32_t             pitchA_v1, 
                                         const uint32_t             pitchB_v8, 
                                         const uint32_t             pitchdst_v1,
                                         const float* __restrict    C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    
    for (uint32_t i = 0; i < proc_dims_v8.y; ++i) {
        B_dex = 0;
        dst_dex = i * pitchdst_v1;
        for (uint32_t j = 0; j < proc_dims_v8.x / 2; ++j) {
            decx::blas::CPUK::GEMM_fp32_dp_kernel<true, _ABC>(A + A_dex, B + B_dex, dst + dst_dex, fmgrL, pitchB_v8, C + dst_dex);
            B_dex += pitchB_v8 * 16;
            dst_dex += 16;
        }
        if (proc_dims_v8.x % 2) {
            decx::blas::CPUK::GEMM_fp32_dp_kernel<false, _ABC>(A + A_dex, B + B_dex, dst + dst_dex, fmgrL, 0, C + dst_dex);
        }
        A_dex += pitchA_v1;
    }
}


template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_fp32_kernel(const float* __restrict                  A, 
                                   const float* __restrict                  B,
                                   float* __restrict                        dst, 
                                   const decx::blas::GEMM_blocking_config*  config,
                                   const uint32_t                           pitchA_v1, 
                                   const uint32_t                           pitchB_v8, 
                                   const uint32_t                           pitchdst_v1,
                                   const float* __restrict                  C)
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
            decx::blas::CPUK::GEMM_fp32_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                proc_dims, &config->_fmgr_L, pitchA_v1, pitchB_v8, pitchdst_v1, C + dst_dex);

            B_dex += config->_fmgr_W.frag_len * pitchB_v8 * 8;
            dst_dex += config->_fmgr_W.frag_len * 8;
        }

        proc_dims.x = config->_fmgr_W.last_frag_len;
        decx::blas::CPUK::GEMM_fp32_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
            proc_dims, &config->_fmgr_L, pitchA_v1, pitchB_v8, pitchdst_v1, C + dst_dex);

        A_dex += config->_fmgr_H.frag_len * pitchA_v1;
    }
}


#endif
