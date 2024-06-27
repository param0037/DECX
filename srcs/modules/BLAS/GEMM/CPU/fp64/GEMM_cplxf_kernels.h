/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GEMM_CPLXF_KERNEL_H_
#define _GEMM_CPLXF_KERNEL_H_

#include "../../../../core/basic.h"
#include "../../../../DSP/CPU_cpf32_avx.h"
#include "../../../../DSP/cplxf_SSE.h"


namespace decx
{
namespace blas {
    namespace CPUK 
    {
        // [C11 C12]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen1x2(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);

        // [C11 C21]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen2x1(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const uint32_t pitchA_v1, const uint32_t pitchdst_v1,
            const bool _first = false, const double* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen1x1(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const bool _first = false, const double* __restrict C = NULL);


        // [C11 C12; C21 C22]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxf_dp_kernel_strassen2x2(const double* __restrict A_line, const double* __restrict B_lane,
            double* __restrict dst, const uint32_t _linear, const uint32_t pitchA_v1, const uint32_t pitchdst_v1,
            const bool _first = false, const double* __restrict C = NULL);

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
GEMM_cplxf_dp_kernel_strassen1x2(const double* __restrict A_line,   const double* __restrict B_lane,
                                double* __restrict dst,             const uint32_t _linear, 
                                const bool _first,                  const double* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu[2];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm256_load_pd(dst);                 _accu[1]._vd = _mm256_load_pd(dst + 4);
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            __m256d recv0 = _mm256_load_pd(C);                  __m256d recv1 = _mm256_load_pd(C + 4);
            recv0 = _mm256_permute4x64_pd(recv0, 0b11011000);   recv1 = _mm256_permute4x64_pd(recv1, 0b11011000);
            _accu[0]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x20);
            _accu[1]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x31);
        }
        else {
            _accu[0]._vd = _mm256_setzero_pd();             _accu[1]._vd = _mm256_setzero_pd();
        }
    }

    /**
    * The pitch of matrix A allows access of data where row address is width + 1 
    * if width is not aligned to 4 in de::CPf datatype.
    */
    for (uint32_t i = 0; i < _L_v2; ++i)
    {
        __m256d A_row0 = _mm256_castpd128_pd256(_mm_load_pd(A_line + i * 2));
        A_row0 = _mm256_insertf128_pd(A_row0, _mm256_castpd256_pd128(A_row0), 1);

        __m256 A11 = _mm256_castpd_ps(_mm256_permute_pd(A_row0, 0b0000));
        __m256 A12 = _mm256_castpd_ps(_mm256_permute_pd(A_row0, 0b1111));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm256_setzero_ps();
        }

        __m256 B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex));
        __m256 B2 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 12));

        __m256 tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(B1, B2), A11);   // M1
        _accu[0]._vf = _mm256_add_ps(_accu[0]._vf, tmp);        // C11 += M1

        // M2 = 0   nop
        B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 8));      // B21
        // M4 = 0   nop

        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(B1, B2), A12);       // M7                                             // M7
        _accu[0]._vf = _mm256_add_ps(_accu[0]._vf, tmp);        // C11 += M7

        B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 4));      // B12
        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(A11, _mm256_sub_ps(B1, B2));       // M3
        _accu[1]._vf = _mm256_add_ps(_accu[1]._vf, tmp);        // C12 += M3

        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(A11, A12), B2);      // M5
        _accu[0]._vf = _mm256_sub_ps(_accu[0]._vf, tmp);        // C11 -= M5
        _accu[1]._vf = _mm256_add_ps(_accu[1]._vf, tmp);        // C12 += M5

        // M6 != 0, but C22 doen't exist, so nop

        B_dex += 16;
    }

    _mm256_store_pd(dst, _accu[0]._vd);                 _mm256_store_pd(dst + 4, _accu[1]._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_strassen2x1(const double* __restrict A_line,       const double* __restrict B_lane,
                                 double* __restrict dst,                const uint32_t _linear,
                                 const uint32_t pitchA_v1,              const uint32_t pitchdst_v1,            
                                 const bool _first,                     const double* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm128_reg _accu[4];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm_load_pd(dst);                 _accu[1]._vd = _mm_load_pd(dst + 2);
        _accu[2]._vd = _mm_load_pd(dst + pitchdst_v1);   _accu[3]._vd = _mm_load_pd(dst + pitchdst_v1 + 2);
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            // First row
            __m256d recv = _mm256_load_pd(C);
            recv = _mm256_permute4x64_pd(recv, 0b11011000);
            _accu[0]._vd = _mm256_castpd256_pd128(recv); _accu[1]._vd = _mm256_extractf128_pd(recv, 1);
            // Second row
            recv = _mm256_load_pd(C + pitchdst_v1);
            recv = _mm256_permute4x64_pd(recv, 0b11011000);
            _accu[2]._vd = _mm256_castpd256_pd128(recv); _accu[3]._vd = _mm256_extractf128_pd(recv, 1);
        }
        else {
            _accu[0]._vd = _mm_setzero_pd();             _accu[1]._vd = _mm_setzero_pd();
            _accu[2]._vd = _mm_setzero_pd();             _accu[3]._vd = _mm_setzero_pd();
        }
    }

    /**
    * The pitch of matrix A allows access of data where row address is width + 1 
    * if width is not aligned to 4 in de::CPf datatype.
    */
    for (uint32_t i = 0; i < _L_v2; ++i)
    {
        __m128d A_row0 = _mm_load_pd(A_line + i * 2);
        __m128d A_row1 = _mm_load_pd(A_line + i * 2 + pitchA_v1);

        __m128 A11 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b00));
        __m128 A12 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b11));
        __m128 A21 = _mm_castpd_ps(_mm_permute_pd(A_row1, 0b00));
        __m128 A22 = _mm_castpd_ps(_mm_permute_pd(A_row1, 0b11));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm_setzero_ps();
            A22 = _mm_setzero_ps();
        }

        __m128 B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex));
        __m128 B2 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 10));

        __m128 tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2),
                                                        _mm_add_ps(A11, A22));   // M1
        _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M1
        _accu[3]._vf = _mm_add_ps(_accu[3]._vf, tmp);        // C22 += M1

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(B1, _mm_add_ps(A21, A22));      // M2
        _accu[2]._vf = _mm_add_ps(_accu[2]._vf, tmp);        // C21 += M2
        _accu[3]._vf = _mm_sub_ps(_accu[3]._vf, tmp);        // C22 -= M2

        B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 8));      // B21
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(A22, _mm_sub_ps(B1,
                _mm_castpd_ps(_mm_load_pd(B_lane + B_dex))));                     // M4
        _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C21 += M4
        _accu[2]._vf = _mm_add_ps(_accu[2]._vf, tmp);        // C22 += M4

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2),
                _mm_sub_ps(A12, A22));                                               // M7
        _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M7

        B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 2));      // B12
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(A11, _mm_sub_ps(B1, B2));       // M3
        _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M3
        _accu[3]._vf = _mm_add_ps(_accu[3]._vf, tmp);        // C22 += M3

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(A11, A12), B2);      // M5
        _accu[0]._vf = _mm_sub_ps(_accu[0]._vf, tmp);        // C11 -= M5
        _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M5

        B2 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex));          // B11
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B2, B1),
                _mm_sub_ps(A21, A11));                                               // M6
        _accu[3]._vf = _mm_add_ps(_accu[3]._vf, tmp);        // C22 += M6

        B_dex += 16;
    }

    _mm_store_pd(dst, _accu[0]._vd);                 _mm_store_pd(dst + 2, _accu[1]._vd);
    _mm_store_pd(dst + pitchdst_v1, _accu[2]._vd);   _mm_store_pd(dst + pitchdst_v1 + 2, _accu[3]._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_strassen1x1(const double* __restrict A_line,       const double* __restrict B_lane,
                                 double* __restrict dst,                const uint32_t _linear,       
                                 const bool _first,                     const double* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm128_reg _accu[2];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm_load_pd(dst);                 _accu[1]._vd = _mm_load_pd(dst + 2);
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            __m256d recv = _mm256_load_pd(C);
            recv = _mm256_permute4x64_pd(recv, 0b11011000);
            _accu[0]._vd = _mm256_castpd256_pd128(recv); _accu[1]._vd = _mm256_extractf128_pd(recv, 1);
        }
        else {
            _accu[0]._vd = _mm_setzero_pd();             _accu[1]._vd = _mm_setzero_pd();
        }
    }

    /**
    * The pitch of matrix A allows access of data where row address is width + 1 
    * if width is not aligned to 4 in de::CPf datatype.
    */
    for (uint32_t i = 0; i < _L_v2; ++i)
    {
        __m128d A_row0 = _mm_load_pd(A_line + i * 2);

        __m128 A11 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b00));
        __m128 A12 = _mm_castpd_ps(_mm_permute_pd(A_row0, 0b11));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm_setzero_ps();
        }

        __m128 B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex));
        __m128 B2 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 10));

        __m128 tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2), A11);   // M1
        _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M1

        // M2 = 0   nop
        B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 8));      // B21
        // M4 = 0   nop

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(B1, B2), A12);      // M7
        _accu[0]._vf = _mm_add_ps(_accu[0]._vf, tmp);        // C11 += M7

        //B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 2));      // B12 = 0
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(A11, decx::utils::simd::_mm_signinv_ps(B2)); // M3
        _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M3

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp32(_mm_add_ps(A11, A12), B2);      // M5
        _accu[0]._vf = _mm_sub_ps(_accu[0]._vf, tmp);        // C11 -= M5
        _accu[1]._vf = _mm_add_ps(_accu[1]._vf, tmp);        // C12 += M5

        // C22 dosen't exist    nop

        B_dex += 16;
    }

    _mm_store_pd(dst, _accu[0]._vd);                 _mm_store_pd(dst + 2, _accu[1]._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxf_dp_kernel_strassen2x2(const double* __restrict A_line,       const double* __restrict B_lane,
                                 double* __restrict dst,                const uint32_t _linear,
                                 const uint32_t pitchA_v1,              const uint32_t pitchdst_v1,            
                                 const bool _first,                     const double* __restrict  C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu[4];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm256_load_pd(dst);                 _accu[1]._vd = _mm256_load_pd(dst + 4);
        _accu[2]._vd = _mm256_load_pd(dst + pitchdst_v1);   _accu[3]._vd = _mm256_load_pd(dst + pitchdst_v1 + 4);
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            // First row
            __m256d recv0 = _mm256_load_pd(C);                  __m256d recv1 = _mm256_load_pd(C + 4);
            recv0 = _mm256_permute4x64_pd(recv0, 0b11011000);   recv1 = _mm256_permute4x64_pd(recv1, 0b11011000);
            _accu[0]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x20);
            _accu[1]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x31);
            // Second row
            recv0 = _mm256_load_pd(C + pitchdst_v1);            recv1 = _mm256_load_pd(C + pitchdst_v1 + 4);
            recv0 = _mm256_permute4x64_pd(recv0, 0b11011000);   recv1 = _mm256_permute4x64_pd(recv1, 0b11011000);
            _accu[2]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x20);
            _accu[3]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x31);
        }
        else {
            _accu[0]._vd = _mm256_setzero_pd();             _accu[1]._vd = _mm256_setzero_pd();
            _accu[2]._vd = _mm256_setzero_pd();             _accu[3]._vd = _mm256_setzero_pd();
        }
    }

    /**
    * The pitch of matrix A allows access of data where row address is width + 1 
    * if width is not aligned to 4 in de::CPf datatype.
    */
    for (uint32_t i = 0; i < _L_v2; ++i)
    {
        __m256d A_row0 = _mm256_castpd128_pd256(_mm_load_pd(A_line + i * 2));
        A_row0 = _mm256_insertf128_pd(A_row0, _mm256_castpd256_pd128(A_row0), 1);
        __m256d A_row1 = _mm256_castpd128_pd256(_mm_load_pd(A_line + i * 2 + pitchA_v1));
        A_row1 = _mm256_insertf128_pd(A_row1, _mm256_castpd256_pd128(A_row1), 1);

        __m256 A11 = _mm256_castpd_ps(_mm256_permute_pd(A_row0, 0b0000));
        __m256 A12 = _mm256_castpd_ps(_mm256_permute_pd(A_row0, 0b1111));
        __m256 A21 = _mm256_castpd_ps(_mm256_permute_pd(A_row1, 0b0000));
        __m256 A22 = _mm256_castpd_ps(_mm256_permute_pd(A_row1, 0b1111));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm256_setzero_ps();
            A22 = _mm256_setzero_ps();
        }

        __m256 B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex));
        __m256 B2 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 12));

        __m256 tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(B1, B2),
                                                        _mm256_add_ps(A11, A22));   // M1
        _accu[0]._vf = _mm256_add_ps(_accu[0]._vf, tmp);        // C11 += M1
        _accu[3]._vf = _mm256_add_ps(_accu[3]._vf, tmp);        // C22 += M1

        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(B1, _mm256_add_ps(A21, A22));      // M2
        _accu[2]._vf = _mm256_add_ps(_accu[2]._vf, tmp);        // C21 += M2
        _accu[3]._vf = _mm256_sub_ps(_accu[3]._vf, tmp);        // C22 -= M2

        B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 8));      // B21
        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(A22, _mm256_sub_ps(B1,
            _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex))));                     // M4
        _accu[0]._vf = _mm256_add_ps(_accu[0]._vf, tmp);        // C21 += M4
        _accu[2]._vf = _mm256_add_ps(_accu[2]._vf, tmp);        // C22 += M4

        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(B1, B2),
            _mm256_sub_ps(A12, A22));                                               // M7
        _accu[0]._vf = _mm256_add_ps(_accu[0]._vf, tmp);        // C11 += M7

        B1 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex + 4));      // B12
        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(A11, _mm256_sub_ps(B1, B2));       // M3
        _accu[1]._vf = _mm256_add_ps(_accu[1]._vf, tmp);        // C12 += M3
        _accu[3]._vf = _mm256_add_ps(_accu[3]._vf, tmp);        // C22 += M3

        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(A11, A12), B2);      // M5
        _accu[0]._vf = _mm256_sub_ps(_accu[0]._vf, tmp);        // C11 -= M5
        _accu[1]._vf = _mm256_add_ps(_accu[1]._vf, tmp);        // C12 += M5

        B2 = _mm256_castpd_ps(_mm256_load_pd(B_lane + B_dex));          // B11
        tmp = decx::dsp::CPUK::_cp4_mul_cp4_fp32(_mm256_add_ps(B2, B1),
            _mm256_sub_ps(A21, A11));                                               // M6
        _accu[3]._vf = _mm256_add_ps(_accu[3]._vf, tmp);        // C22 += M6

        B_dex += 16;
    }

    _mm256_store_pd(dst, _accu[0]._vd);                 _mm256_store_pd(dst + 4, _accu[1]._vd);
    _mm256_store_pd(dst + pitchdst_v1, _accu[2]._vd);   _mm256_store_pd(dst + pitchdst_v1 + 4, _accu[3]._vd);
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxf_block_kernel(const double* __restrict A,               const double* __restrict B, 
                                         double* __restrict dst,                    const uint2 proc_dims_v4,
                                         const decx::utils::frag_manager* fmgrL,    const uint32_t pitchA_v1, 
                                         const uint32_t Llen,                       const uint32_t pitchdst_v1,
                                         const double* __restrict C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    const uint32_t _H_v2 = proc_dims_v4.y / 2;

    for (uint32_t k = 0; k < fmgrL->frag_num; ++k) 
    {
        const uint32_t _L_frag = k == fmgrL->frag_num - 1 ? fmgrL->last_frag_len : fmgrL->frag_len;
        A_dex = fmgrL->frag_len * k;

        for (uint32_t i = 0; i < _H_v2; ++i) 
        {
            B_dex = fmgrL->frag_len * k * 8;
            dst_dex = i * pitchdst_v1 * 2;
            for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen2x2<_ABC>(A + A_dex, B + B_dex,
                        dst + dst_dex, _L_frag, pitchA_v1, pitchdst_v1, k == 0, C + dst_dex);

                B_dex += Llen * 8;
                dst_dex += 8;
            }
            if (proc_dims_v4.x % 2) {
                // strassen2x1
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen2x1<_ABC>(A + A_dex, B + B_dex,
                        dst + dst_dex, _L_frag, pitchA_v1, pitchdst_v1, k == 0, C + dst_dex);
            }
            A_dex += pitchA_v1 * 2;
        }
        if (proc_dims_v4.y & 1) {
            B_dex = fmgrL->frag_len * k * 8;
            dst_dex = _H_v2 * pitchdst_v1 * 2;
            for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen1x2<_ABC>(A + A_dex, B + B_dex,
                    dst + dst_dex, _L_frag, k == 0, C + dst_dex);

                B_dex += Llen * 8;
                dst_dex += 8;
            }
            if (proc_dims_v4.x % 2) {
                // strassen1x1
                decx::blas::CPUK::GEMM_cplxf_dp_kernel_strassen1x1<_ABC>(A + A_dex, B + B_dex,
                    dst + dst_dex, _L_frag, k == 0, C + dst_dex);
            }
        }
    }

    for (uint32_t i = 0; i < proc_dims_v4.y; ++i) {
        dst_dex = i * pitchdst_v1;
        for (uint32_t j = 0; j < proc_dims_v4.x / 2; ++j) {
            __m256d reg0 = _mm256_load_pd(dst + dst_dex);
            __m256d reg1 = _mm256_load_pd(dst + dst_dex + 4);
            __m256d tmp0 = _mm256_shuffle_pd(reg0, reg1, 0b1100);
            __m256d tmp1 = _mm256_shuffle_pd(reg0, reg1, 0b0011);
            reg0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
            reg1 = _mm256_permute2f128_pd(tmp0, tmp1, 0x13);
            _mm256_store_pd(dst + dst_dex, reg0);         _mm256_store_pd(dst + dst_dex + 4, reg1);
            
            dst_dex += 8;
        }
        if (proc_dims_v4.x & 1) {
            __m256d reg = _mm256_load_pd(dst + dst_dex);
            reg = _mm256_permute4x64_pd(reg, 0b11011000);
            _mm256_store_pd(dst + dst_dex, reg);
        }
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxf_kernel(const double* __restrict A,     const double* __restrict B,
                                    double* __restrict dst,         const decx::blas::GEMM_blocking_config* config,
                                    const uint32_t pitchA_v1,       const uint32_t Llen, 
                                    const uint32_t pitchdst_v1,     const double* __restrict C)
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
