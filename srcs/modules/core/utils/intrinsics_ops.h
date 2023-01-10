/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _INTRINSICS_OPS_H_
#define _INTRINSICS_OPS_H_


#include "decx_utils_macros.h"
#include "../vector_defines.h"


namespace decx
{
    namespace utils
    {
#ifdef _DECX_CPU_CODES_
        namespace simd {
            typedef union xmm128_reg {
                __m128 _vf;
                __m128d _vd;
                __m128i _vi;
            }_mmv128;


            typedef union xmm256_reg {
                __m256 _vf;
                __m256d _vd;
                __m256i _vi;
            }_mmv256;


            static float _mm128_h_sum(__m128 v);


            static float _mm256_h_sum(__m256 v);


            static double _mm256d_h_sum(__m256d v);


            /**
            * The data move from higher address to lower address for 1 element
            * @param __proc : the pointer of the value to be processed
            */
            FORCEINLINE __m256 _mm256_shift1_H2L(__m256 __proc);


            /**
            * The data move from higher address to lower address for 2 elements
            * @param __proc : the pointer of the value to be processed
            */
            FORCEINLINE __m256 _mm256_shift2_H2L(__m256 __proc);


            /**
            * The data move from higher address to lower address for 1 element
            * @param __proc : the pointer of the value to be processed
            */
            FORCEINLINE __m256 _mm256_shift1_L2H(__m256 __proc);


            /**
            * The data move from higher address to lower address for 2 elements
            * @param __proc : the pointer of the value to be processed
            */
            FORCEINLINE __m256 _mm256_shift2_L2H(__m256 __proc);



            FORCEINLINE __m256 _mm256_abs_ps(__m256 __proc);
        }
#endif
    }
}


#ifdef _DECX_CPU_CODES_
static float decx::utils::simd::_mm128_h_sum(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}


static double decx::utils::simd::_mm256d_h_sum(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}



static float decx::utils::simd::_mm256_h_sum(__m256 x) {
    __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    __m128 loQuad = _mm256_castps256_ps128(x);
    __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    hiQuad = _mm_shuffle_ps(sumQuad, sumQuad, _MM_SHUFFLE(2, 3, 0, 1));
    sumQuad = _mm_add_ps(sumQuad, hiQuad);
    loQuad = _mm_movehl_ps(hiQuad, sumQuad);
    sumQuad = _mm_add_ss(sumQuad, loQuad);
    return _mm_cvtss_f32(sumQuad);
}


FORCEINLINE 
__m256 decx::utils::simd::_mm256_shift1_H2L(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(0, 3, 2, 1));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 81);
    return _mm256_blend_ps(tmp0, tmp1, 0x88);
}


FORCEINLINE __m256 decx::utils::simd::_mm256_shift2_H2L(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 81);
    return _mm256_blend_ps(tmp0, tmp1, 0xcc);
}


FORCEINLINE __m256 decx::utils::simd::_mm256_shift1_L2H(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(2, 1, 0, 3));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 41);
    return _mm256_blend_ps(tmp0, tmp1, 0x11);
}


FORCEINLINE __m256 decx::utils::simd::_mm256_shift2_L2H(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 41);
    return _mm256_blend_ps(tmp0, tmp1, 0x33);
}



FORCEINLINE __m256 decx::utils::simd::_mm256_abs_ps(__m256 __proc)
{
    return _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(__proc), _mm256_set1_epi32(0x7fffffff)));
}



// -------------------------------------- _mm256d ---------------------------------------


namespace decx
{
    namespace utils
    {
#ifdef _DECX_CPU_CODES_
        namespace simd {
            FORCEINLINE __m256d _mm256d_shift1_H2L(__m256d _proc);
        }
#endif
    }
}


FORCEINLINE 
__m256d decx::utils::simd::_mm256d_shift1_H2L(__m256d _proc) {
    __m256d tmp = _mm256_permute_pd(_proc, 0b0101);
    _proc = _mm256_permute2f128_pd(tmp, tmp, 1);
    return _mm256_blend_pd(tmp, _proc, 0b1010);
}


#endif      // #ifdef _DECX_CPU_CODES_


#endif