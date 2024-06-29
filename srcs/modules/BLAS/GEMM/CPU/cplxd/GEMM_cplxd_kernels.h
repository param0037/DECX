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


#ifndef _GEMM_CPLXD_KERNEL_H_
#define _GEMM_CPLXD_KERNEL_H_

#include "../../../../core/basic.h"
#include "../../../../DSP/CPU_cpd64_avx.h"
#include "../../../../DSP/cplxd_SSE.h"


namespace decx
{
namespace blas {
    namespace CPUK 
    {
        // [C11 C12]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxd_dp_kernel_strassen1x2(const de::CPd* __restrict A_line, const de::CPd* __restrict B_lane,
            de::CPd* __restrict dst, const uint32_t _linear, const bool _first = false, const de::CPd* __restrict C = NULL);

        // [C11 C21]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxd_dp_kernel_strassen2x1(const de::CPd* __restrict A_line, const de::CPd* __restrict B_lane,
            de::CPd* __restrict dst, const uint32_t _linear, const uint32_t pitchA_v1, const uint32_t pitchdst_v1,
            const bool _first = false, const de::CPd* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxd_dp_kernel_strassen1x1(const de::CPd* __restrict A_line, const de::CPd* __restrict B_lane,
            de::CPd* __restrict dst, const uint32_t _linear, const bool _first = false, const de::CPd* __restrict C = NULL);


        // [C11 C12; C21 C22]
        template <bool _ABC>
        static _THREAD_CALL_ void GEMM_cplxd_dp_kernel_strassen2x2(const de::CPd* __restrict A_line, const de::CPd* __restrict B_lane,
            de::CPd* __restrict dst, const uint32_t _linear, const uint32_t pitchA_v1, const uint32_t pitchdst_v1,
            const bool _first = false, const de::CPd* __restrict C = NULL);

        /**
        * The layout of dst and C should be completely consistant. Normally it will be, by the definition of GEMM.
        */
        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_cplxd_block_kernel(const de::CPd* __restrict A, const de::CPd* __restrict B,
            de::CPd* __restrict dst, const uint2 proc_dims_v8, const decx::utils::frag_manager* fmgrL,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const de::CPd* __restrict C = NULL);


        template <bool _ABC>
        static _THREAD_FUNCTION_ void GEMM_cplxd_kernel(const de::CPd* __restrict A, const de::CPd* __restrict B,
            de::CPd* __restrict dst, const decx::blas::GEMM_blocking_config* config,
            const uint32_t pitchA_v1, const uint32_t Llen, const uint32_t pitchdst_v1, const de::CPd* __restrict C = NULL);
    }
}
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxd_dp_kernel_strassen1x2(const de::CPd* __restrict A_line,   const de::CPd* __restrict B_lane,
                                de::CPd* __restrict dst,             const uint32_t _linear, 
                                const bool _first,                  const de::CPd* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu[2];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm256_load_pd((double*)dst);            _accu[1]._vd = _mm256_load_pd((double*)(dst + 2));
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            __m256d recv0 = _mm256_load_pd((double*)(C));       __m256d recv1 = _mm256_load_pd((double*)(C + 2));
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
        __m256d A_row0 = _mm256_load_pd((double*)(A_line + i * 2));
        __m256d A11 = _mm256_permute2f128_pd(A_row0, A_row0, 0x20);
        __m256d A12 = _mm256_permute2f128_pd(A_row0, A_row0, 0x31);

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm256_setzero_pd();
        }

        __m256d B1 = _mm256_load_pd((double*)(B_lane + B_dex));
        __m256d B2 = _mm256_load_pd((double*)(B_lane + B_dex + 6));

        __m256d tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(B1, B2), A11);   // M1
        _accu[0]._vd = _mm256_add_pd(_accu[0]._vd, tmp);        // C11 += M1

        // M2 = 0   nop
        B1 = _mm256_load_pd((double*)(B_lane + B_dex + 4));      // B21
        // M4 = 0   nop

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(B1, B2), A12);       // M7                                             // M7
        _accu[0]._vd = _mm256_add_pd(_accu[0]._vd, tmp);        // C11 += M7

        B1 = _mm256_load_pd((double*)(B_lane + B_dex + 2));      // B12
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(A11, _mm256_sub_pd(B1, B2));       // M3
        _accu[1]._vd = _mm256_add_pd(_accu[1]._vd, tmp);        // C12 += M3

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(A11, A12), B2);      // M5
        _accu[0]._vd = _mm256_sub_pd(_accu[0]._vd, tmp);        // C11 -= M5
        _accu[1]._vd = _mm256_add_pd(_accu[1]._vd, tmp);        // C12 += M5

        // M6 != 0, but C22 doen't exist, so nop

        B_dex += 8;
    }

    _mm256_store_pd((double*)dst, _accu[0]._vd);     _mm256_store_pd((double*)(dst + 2), _accu[1]._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxd_dp_kernel_strassen2x1(const de::CPd* __restrict A_line,      const de::CPd* __restrict B_lane,
                                 de::CPd* __restrict dst,               const uint32_t _linear,
                                 const uint32_t pitchA_v1,              const uint32_t pitchdst_v1,            
                                 const bool _first,                     const de::CPd* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm128_reg _accu[4];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm_load_pd((double*)dst);                   
        _accu[1]._vd = _mm_load_pd((double*)(dst + 1));
        _accu[2]._vd = _mm_load_pd((double*)(dst + pitchdst_v1));   
        _accu[3]._vd = _mm_load_pd((double*)(dst + pitchdst_v1 + 1));
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            // First row
            __m256d recv = _mm256_load_pd((double*)C);
            _accu[0]._vd = _mm256_castpd256_pd128(recv); _accu[1]._vd = _mm256_extractf128_pd(recv, 1);
            // Second row
            recv = _mm256_load_pd((double*)(C + pitchdst_v1));
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
        __m128d A11 = _mm_load_pd((double*)(A_line + i * 2));
        __m128d A12 = _mm_load_pd((double*)(A_line + i * 2 + 1));
        __m128d A21 = _mm_load_pd((double*)(A_line + i * 2 + pitchA_v1));
        __m128d A22 = _mm_load_pd((double*)(A_line + i * 2 + 1 + pitchA_v1));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm_setzero_pd();
            A22 = _mm_setzero_pd();
        }

        __m128d B1 = _mm_load_pd((double*)(B_lane + B_dex));
        __m128d B2 = _mm_load_pd((double*)(B_lane + B_dex + 5));

        __m128d tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(B1, B2),
                                                        _mm_add_pd(A11, A22));   // M1
        _accu[0]._vd = _mm_add_pd(_accu[0]._vd, tmp);        // C11 += M1
        _accu[3]._vd = _mm_add_pd(_accu[3]._vd, tmp);        // C22 += M1

        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(B1, _mm_add_pd(A21, A22));      // M2
        _accu[2]._vd = _mm_add_pd(_accu[2]._vd, tmp);        // C21 += M2
        _accu[3]._vd = _mm_sub_pd(_accu[3]._vd, tmp);        // C22 -= M2

        B1 = _mm_load_pd((double*)(B_lane + B_dex + 4));      // B21
        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(A22, _mm_sub_pd(B1,
                _mm_load_pd((double*)(B_lane + B_dex))));                     // M4
        _accu[0]._vd = _mm_add_pd(_accu[0]._vd, tmp);        // C21 += M4
        _accu[2]._vd = _mm_add_pd(_accu[2]._vd, tmp);        // C22 += M4

        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(B1, B2),
                _mm_sub_pd(A12, A22));                                               // M7
        _accu[0]._vd = _mm_add_pd(_accu[0]._vd, tmp);        // C11 += M7

        B1 = _mm_load_pd((double*)(B_lane + B_dex + 1));      // B12
        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(A11, _mm_sub_pd(B1, B2));       // M3
        _accu[1]._vd = _mm_add_pd(_accu[1]._vd, tmp);        // C12 += M3
        _accu[3]._vd = _mm_add_pd(_accu[3]._vd, tmp);        // C22 += M3

        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(A11, A12), B2);      // M5
        _accu[0]._vd = _mm_sub_pd(_accu[0]._vd, tmp);        // C11 -= M5
        _accu[1]._vd = _mm_add_pd(_accu[1]._vd, tmp);        // C12 += M5

        B2 = _mm_load_pd((double*)(B_lane + B_dex));          // B11
        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(B2, B1),
                _mm_sub_pd(A21, A11));                                               // M6
        _accu[3]._vd = _mm_add_pd(_accu[3]._vd, tmp);        // C22 += M6

        B_dex += 8;
    }

    _mm_store_pd((double*)dst, _accu[0]._vd);        
    _mm_store_pd((double*)(dst + 1), _accu[1]._vd);
    _mm_store_pd((double*)(dst + pitchdst_v1), _accu[2]._vd);   
    _mm_store_pd((double*)(dst + pitchdst_v1 + 1), _accu[3]._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxd_dp_kernel_strassen1x1(const de::CPd* __restrict A_line,      const de::CPd* __restrict B_lane,
                                 de::CPd* __restrict dst,               const uint32_t _linear,       
                                 const bool _first,                     const de::CPd* __restrict C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm128_reg _accu[2];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm_load_pd((double*)dst);        _accu[1]._vd = _mm_load_pd((double*)(dst + 1));
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            __m256d recv = _mm256_load_pd((double*)C);
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
        __m128d A11 = _mm_load_pd((double*)(A_line + i * 2));
        __m128d A12 = _mm_load_pd((double*)(A_line + i * 2 + 1));

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm_setzero_pd();
        }

        __m128d B1 = _mm_load_pd((double*)(B_lane + B_dex));
        __m128d B2 = _mm_load_pd((double*)(B_lane + B_dex + 5));

        __m128d tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(B1, B2), A11);   // M1
        _accu[0]._vd = _mm_add_pd(_accu[0]._vd, tmp);        // C11 += M1

        // M2 = 0   nop
        B1 = _mm_load_pd((double*)(B_lane + B_dex + 4));      // B21
        // M4 = 0   nop

        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(B1, B2), A12);      // M7
        _accu[0]._vd = _mm_add_pd(_accu[0]._vd, tmp);        // C11 += M7

        //B1 = _mm_castpd_ps(_mm_load_pd(B_lane + B_dex + 2));      // B12 = 0
        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(A11, decx::utils::simd::_mm_signinv_pd(B2)); // M3
        _accu[1]._vd = _mm_add_pd(_accu[1]._vd, tmp);        // C12 += M3

        tmp = decx::dsp::CPUK::_cp_mul_cp_fp64(_mm_add_pd(A11, A12), B2);      // M5
        _accu[0]._vd = _mm_sub_pd(_accu[0]._vd, tmp);        // C11 -= M5
        _accu[1]._vd = _mm_add_pd(_accu[1]._vd, tmp);        // C12 += M5

        // C22 dosen't exist    nop

        B_dex += 8;
    }

    _mm_store_pd((double*)dst, _accu[0]._vd);         _mm_store_pd((double*)(dst + 1), _accu[1]._vd);
}



template <bool _ABC>
static _THREAD_CALL_ void decx::blas::CPUK::
GEMM_cplxd_dp_kernel_strassen2x2(const de::CPd* __restrict A_line,       const de::CPd* __restrict B_lane,
                                 de::CPd* __restrict dst,                const uint32_t _linear,
                                 const uint32_t pitchA_v1,              const uint32_t pitchdst_v1,            
                                 const bool _first,                     const de::CPd* __restrict  C)
{
    uint32_t B_dex = 0;
    decx::utils::simd::xmm256_reg _accu[4];
    const uint32_t _L_v2 = decx::utils::fast_uint_ceil2<uint32_t>(_linear);

    if (!_first) {
        _accu[0]._vd = _mm256_load_pd((double*)(dst));                 
        _accu[1]._vd = _mm256_load_pd((double*)(dst + 2));
        _accu[2]._vd = _mm256_load_pd((double*)(dst + pitchdst_v1));   
        _accu[3]._vd = _mm256_load_pd((double*)(dst + pitchdst_v1 + 2));
    }
    else {
        if constexpr (_ABC) {
            /**
            * Since matrix C is layouted as normal, in Strassen's Algorithm for avx2 (4x cplxf),
            * the data has to be rearranged to the same form (layout) as that in dst (and _accu registers).
            */
            // First row
            __m256d recv0 = _mm256_load_pd((double*)(C));       __m256d recv1 = _mm256_load_pd((double*)(C + 2));
            _accu[0]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x20);
            _accu[1]._vd = _mm256_permute2f128_pd(recv0, recv1, 0x31);
            // Second row
            recv0 = _mm256_load_pd((double*)(C + pitchdst_v1)); recv1 = _mm256_load_pd((double*)(C + 2 + pitchdst_v1));
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
        __m256d A_row0 = _mm256_load_pd((double*)(A_line + i * 2));
        __m256d A11 = _mm256_permute2f128_pd(A_row0, A_row0, 0x20);
        __m256d A12 = _mm256_permute2f128_pd(A_row0, A_row0, 0x31);

        A_row0 = _mm256_load_pd((double*)(A_line + i * 2 + pitchA_v1));
        __m256d A21 = _mm256_permute2f128_pd(A_row0, A_row0, 0x20);
        __m256d A22 = _mm256_permute2f128_pd(A_row0, A_row0, 0x31);

        if ((i == _L_v2 - 1) && (_linear & 1)) {
            A12 = _mm256_setzero_pd();
            A22 = _mm256_setzero_pd();
        }

        __m256d B1 = _mm256_load_pd((double*)(B_lane + B_dex));
        __m256d B2 = _mm256_load_pd((double*)(B_lane + B_dex + 6));

        __m256d tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(B1, B2),
                                                         _mm256_add_pd(A11, A22));   // M1
        _accu[0]._vd = _mm256_add_pd(_accu[0]._vd, tmp);        // C11 += M1
        _accu[3]._vd = _mm256_add_pd(_accu[3]._vd, tmp);        // C22 += M1

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(B1, _mm256_add_pd(A21, A22));      // M2
        _accu[2]._vd = _mm256_add_pd(_accu[2]._vd, tmp);        // C21 += M2
        _accu[3]._vd = _mm256_sub_pd(_accu[3]._vd, tmp);        // C22 -= M2

        B1 = _mm256_load_pd((double*)(B_lane + B_dex + 4));      // B21
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(A22, _mm256_sub_pd(B1,
                _mm256_load_pd((double*)(B_lane + B_dex))));                     // M4
        _accu[0]._vd = _mm256_add_pd(_accu[0]._vd, tmp);        // C21 += M4
        _accu[2]._vd = _mm256_add_pd(_accu[2]._vd, tmp);        // C22 += M4

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(B1, B2),
            _mm256_sub_pd(A12, A22));                                               // M7
        _accu[0]._vd = _mm256_add_pd(_accu[0]._vd, tmp);        // C11 += M7

        B1 = _mm256_load_pd((double*)(B_lane + B_dex + 2));      // B12
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(A11, _mm256_sub_pd(B1, B2));       // M3
        _accu[1]._vd = _mm256_add_pd(_accu[1]._vd, tmp);        // C12 += M3
        _accu[3]._vd = _mm256_add_pd(_accu[3]._vd, tmp);        // C22 += M3

        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(A11, A12), B2);      // M5
        _accu[0]._vd = _mm256_sub_pd(_accu[0]._vd, tmp);        // C11 -= M5
        _accu[1]._vd = _mm256_add_pd(_accu[1]._vd, tmp);        // C12 += M5

        B2 = _mm256_load_pd((double*)(B_lane + B_dex));          // B11
        tmp = decx::dsp::CPUK::_cp2_mul_cp2_fp64(_mm256_add_pd(B2, B1),
            _mm256_sub_pd(A21, A11));                                               // M6
        _accu[3]._vd = _mm256_add_pd(_accu[3]._vd, tmp);        // C22 += M6

        B_dex += 8;
    }

    _mm256_store_pd((double*)dst, _accu[0]._vd);
    _mm256_store_pd((double*)(dst + 2), _accu[1]._vd);
    _mm256_store_pd((double*)(dst + pitchdst_v1), _accu[2]._vd);
    _mm256_store_pd((double*)(dst + pitchdst_v1 + 2), _accu[3]._vd);
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxd_block_kernel(const de::CPd* __restrict A,               const de::CPd* __restrict B, 
                                          de::CPd* __restrict dst,                   const uint2 proc_dims_v2,
                                          const decx::utils::frag_manager* fmgrL,    const uint32_t pitchA_v1, 
                                          const uint32_t Llen,                       const uint32_t pitchdst_v1,
                                          const de::CPd* __restrict C)
{
    uint64_t A_dex = 0, B_dex = 0, dst_dex = 0;
    const uint32_t _H_v2 = proc_dims_v2.y / 2;

    for (uint32_t k = 0; k < fmgrL->frag_num; ++k) 
    {
        const uint32_t _L_frag = k == fmgrL->frag_num - 1 ? fmgrL->last_frag_len : fmgrL->frag_len;
        A_dex = fmgrL->frag_len * k;

        for (uint32_t i = 0; i < _H_v2; ++i) 
        {
            B_dex = fmgrL->frag_len * k * 4;
            dst_dex = i * pitchdst_v1 * 2;
            for (uint32_t j = 0; j < proc_dims_v2.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxd_dp_kernel_strassen2x2<_ABC>(A + A_dex, B + B_dex,
                        dst + dst_dex, _L_frag, pitchA_v1, pitchdst_v1, k == 0, C + dst_dex);

                B_dex += Llen * 4;
                dst_dex += 4;
            }
            if (proc_dims_v2.x & 1) {
                // strassen2x1
                decx::blas::CPUK::GEMM_cplxd_dp_kernel_strassen2x1<_ABC>(A + A_dex, B + B_dex,
                        dst + dst_dex, _L_frag, pitchA_v1, pitchdst_v1, k == 0, C + dst_dex);
            }
            A_dex += pitchA_v1 * 2;
        }
        if (proc_dims_v2.y & 1) {
            B_dex = fmgrL->frag_len * k * 4;
            dst_dex = _H_v2 * pitchdst_v1 * 2;
            
            for (uint32_t j = 0; j < proc_dims_v2.x / 2; ++j) {
                decx::blas::CPUK::GEMM_cplxd_dp_kernel_strassen1x2<_ABC>(A + A_dex, B + B_dex,
                    dst + dst_dex, _L_frag, k == 0, C + dst_dex);

                B_dex += Llen * 4;
                dst_dex += 4;
            }
            if (proc_dims_v2.x % 2) {
                // strassen1x1
                decx::blas::CPUK::GEMM_cplxd_dp_kernel_strassen1x1<_ABC>(A + A_dex, B + B_dex,
                    dst + dst_dex, _L_frag, k == 0, C + dst_dex);
            }
        }
    }

    for (uint32_t i = 0; i < proc_dims_v2.y; ++i) {
        dst_dex = i * pitchdst_v1;
        for (uint32_t j = 0; j < proc_dims_v2.x / 2; ++j) 
        {
            __m256d stg0, stg1;
            __m256d recv0 = _mm256_load_pd((double*)(dst + dst_dex));
            __m256d recv1 = _mm256_load_pd((double*)(dst + dst_dex + 2));
            stg0 = _mm256_permute2f128_pd(recv0, recv1, 0x20);
            stg1 = _mm256_permute2f128_pd(recv0, recv1, 0x31);

            _mm256_store_pd((double*)(dst + dst_dex), stg0); 
            _mm256_store_pd((double*)(dst + dst_dex + 2), stg1);
            
            dst_dex += 4;
        }
    }
}



template <bool _ABC>
static _THREAD_FUNCTION_ void 
decx::blas::CPUK::GEMM_cplxd_kernel(const de::CPd* __restrict A,     const de::CPd* __restrict B,
                                    de::CPd* __restrict dst,         const decx::blas::GEMM_blocking_config* config,
                                    const uint32_t pitchA_v1,       const uint32_t Llen, 
                                    const uint32_t pitchdst_v1,     const de::CPd* __restrict C)
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
            decx::blas::CPUK::GEMM_cplxd_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                        proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);

            A_dex += config->_fmgr_H.frag_len * pitchA_v1;
            dst_dex += config->_fmgr_H.frag_len * pitchdst_v1;
        }

        proc_dims.y = config->_fmgr_H.last_frag_len;
        decx::blas::CPUK::GEMM_cplxd_block_kernel<_ABC>(A + A_dex, B + B_dex, dst + dst_dex,
                    proc_dims, &config->_fmgr_L, pitchA_v1, Llen, pitchdst_v1, C + dst_dex);
    }
}



#endif
