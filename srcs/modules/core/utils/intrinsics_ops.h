/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _INTRINSICS_OPS_H_
#define _INTRINSICS_OPS_H_


#include "decx_utils_macros.h"
#include "../vector_defines.h"
#include "simd_fast_math_avx2.h"
#include "simd_fast_math_sse.h"


namespace decx
{
    namespace utils
    {
#ifdef _DECX_CPU_PARTS_
        namespace simd 
        {
            typedef union __align__(8) xmm64_reg {
                __m64 _m64;
                double _fp64;
                uint64_t _ull;
            }_mmv64;


            typedef union __align__(16) xmm128_reg {
                __m128 _vf;
                __m128d _vd;
                __m128i _vi;

                float _arrf[4];
                int32_t _arri[4];
            }_mmv128;


            typedef union __align__(32) xmm256_reg {
                __m256 _vf;
                __m256d _vd;
                __m256i _vi;

                float _arrf[8];
                double _arrd[4];
            }_mmv256;


            static float _mm128_h_sum(__m128 v);


            static float _mm256_h_sum(__m256 v);

            static float _mm256_h_max(__m256 v);

            static float _mm256_h_min(__m256 v);


            static double _mm256d_h_sum(__m256d v);
            static int64_t _mm256i_h_sum_epi64(__m256i v);

            static double _mm256d_h_max(__m256d v);

            static double _mm256d_h_min(__m256d v);

            static uint8_t _mm128_h_max_u8(__m128i v);

            static uint8_t _mm128_h_min_u8(__m128i v);


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



            
        }
#endif
    }
}


#ifdef _DECX_CPU_PARTS_
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


static int64_t decx::utils::simd::_mm256i_h_sum_epi64(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extractf128_si256(v, 1); // high 128
    vlow = _mm_add_epi64(vlow, vhigh);     // reduce down to 128

    __m128i high64 = _mm_unpackhi_epi64(vlow, vlow);
    return _mm_extract_epi64(_mm_add_epi64(vlow, high64), 0);
}



static double decx::utils::simd::_mm256d_h_max(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_max_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_max_sd(vlow, high64));  // reduce to scalar
}


static double decx::utils::simd::_mm256d_h_min(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_min_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_min_sd(vlow, high64));  // reduce to scalar
}


static uint8_t decx::utils::simd::_mm128_h_max_u8(__m128i v) {
    __m128i _halv_2 = _mm_max_epu8(v, _mm_shuffle_epi32(v, 0b01001110));
    __m128i _halv_4 = _mm_max_epu8(_halv_2, _mm_shuffle_epi32(_halv_2, 0b10110001));
    __m128i _halv_8 = _mm_max_epu8(_halv_4, _mm_shuffle_epi8(_halv_4, _mm_set1_epi32(0x01000302)));
    __m128i _halv_16 = _mm_max_epu8(_halv_8, _mm_shuffle_epi8(_halv_8, _mm_set1_epi16(0x0001)));
#ifdef _MSC_VER
    return _halv_16.m128i_u8[0];
#endif
#ifdef __GNUC__
    return *((uint8_t*)&_halv_16);
#endif
}



static uint8_t decx::utils::simd::_mm128_h_min_u8(__m128i v) {
    __m128i _halv_2 = _mm_min_epu8(v, _mm_shuffle_epi32(v, 0b01001110));
    __m128i _halv_4 = _mm_min_epu8(_halv_2, _mm_shuffle_epi32(_halv_2, 0b10110001));
    __m128i _halv_8 = _mm_min_epu8(_halv_4, _mm_shuffle_epi8(_halv_4, _mm_set1_epi32(0x01000302)));
    __m128i _halv_16 = _mm_min_epu8(_halv_8, _mm_shuffle_epi8(_halv_8, _mm_set1_epi16(0x0001)));
#ifdef _MSC_VER
    return _halv_16.m128i_u8[0];
#endif
#ifdef __GNUC__
    return *((uint8_t*)&_halv_16);
#endif
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



static float decx::utils::simd::_mm256_h_max(__m256 x) {
    __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    __m128 loQuad = _mm256_castps256_ps128(x);
    __m128 sumQuad = _mm_max_ps(loQuad, hiQuad);
    hiQuad = _mm_shuffle_ps(sumQuad, sumQuad, _MM_SHUFFLE(2, 3, 0, 1));
    sumQuad = _mm_max_ps(sumQuad, hiQuad);
    loQuad = _mm_movehl_ps(hiQuad, sumQuad);
    sumQuad = _mm_max_ss(sumQuad, loQuad);
    return _mm_cvtss_f32(sumQuad);
}



static float decx::utils::simd::_mm256_h_min(__m256 x) {
    __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    __m128 loQuad = _mm256_castps256_ps128(x);
    __m128 sumQuad = _mm_min_ps(loQuad, hiQuad);
    hiQuad = _mm_shuffle_ps(sumQuad, sumQuad, _MM_SHUFFLE(2, 3, 0, 1));
    sumQuad = _mm_min_ps(sumQuad, hiQuad);
    loQuad = _mm_movehl_ps(hiQuad, sumQuad);
    sumQuad = _mm_min_ss(sumQuad, loQuad);
    return _mm_cvtss_f32(sumQuad);
}



FORCEINLINE __m256 decx::utils::simd::_mm256_shift1_H2L(__m256 __proc)
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




// -------------------------------------- _mm256d ---------------------------------------


namespace decx
{
    namespace utils
    {
#ifdef _DECX_CPU_PARTS_
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


#endif      // #ifdef _DECX_CPU_PARTS_

#endif