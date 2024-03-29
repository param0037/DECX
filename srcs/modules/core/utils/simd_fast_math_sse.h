/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _SIMD_FAST_MATH_SSE_H_
#define _SIMD_FAST_MATH_SSE_H_

#ifdef _DECX_CPU_PARTS_

#include "decx_utils_macros.h"
#include "../vector_defines.h"


namespace decx
{
namespace utils{
    namespace simd
    {
        FORCEINLINE __m128 _mm_abs_ps(__m128 __proc);
        FORCEINLINE __m128 _mm_signinv_ps(__m128 __proc);
        FORCEINLINE __m128 _mm_signinv_ps_masked(__m128 __proc, __m128 __mask);


        FORCEINLINE __m128d _mm_abs_pd(__m128d __proc);
        FORCEINLINE __m128d _mm_signinv_pd(__m128d __proc);

        inline __m128 _mm_atan2_ps(const __m128 __y, const __m128 __x);
        inline __m128 _mm_cos_ps(const __m128 __x);
        inline __m128 _mm_sin_ps(const __m128 __x);
    }
}
}



FORCEINLINE __m128 decx::utils::simd::_mm_abs_ps(__m128 __proc)
{
    return _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(__proc), _mm_set1_epi32(0x7fffffff)));
}


FORCEINLINE __m128 decx::utils::simd::_mm_signinv_ps(__m128 __proc)
{
    return _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(__proc), _mm_set1_epi32(0x80000000)));
}


FORCEINLINE __m128d decx::utils::simd::_mm_abs_pd(__m128d __proc)
{
    return _mm_castsi128_pd(_mm_and_si128(_mm_castpd_si128(__proc), _mm_set1_epi64x(0x7fffffffffffffff)));
}


FORCEINLINE __m128d decx::utils::simd::_mm_signinv_pd(__m128d __proc)
{
    return _mm_castsi128_pd(_mm_xor_si128(_mm_castpd_si128(__proc), _mm_set1_epi64x(0x8000000000000000)));
}


FORCEINLINE __m128 decx::utils::simd::_mm_signinv_ps_masked(__m128 __proc, __m128 __mask)
{
    return _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(__proc),
        _mm_and_si128(_mm_castps_si128(__mask), _mm_set1_epi32(0x80000000))));
}


inline __m128 decx::utils::simd::_mm_atan2_ps(const __m128 __y, const __m128 __x)
{
    __m128 dst;

    const __m128 _ax = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(__x), _mm_set1_epi32(0x7FFFFFFF)));
    const __m128 _ay = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(__y), _mm_set1_epi32(0x7FFFFFFF)));
    const __m128 _a = _mm_div_ps(_mm_min_ps(_ax, _ay), _mm_max_ps(_ax, _ay));
    const __m128 _s = _mm_mul_ps(_a, _a);
    const __m128 _r1 = _mm_fmadd_ps(_s, _mm_set1_ps(-0.0464964749), _mm_set1_ps(0.15931422));
    const __m128 _r2 = _mm_fmadd_ps(_r1, _s, _mm_set1_ps(-0.327622764));
    const __m128 _r3 = _mm_mul_ps(_a, _mm_fmadd_ps(_r2, _s, _mm_set1_ps(1)));
    dst = _r3;
    __m128 _crit = _mm_cmp_ps(_ay, _ax, _CMP_GT_OS);
    dst = _mm_blendv_ps(dst, _mm_sub_ps(_mm_set1_ps(1.57079637f), dst), _crit);
    _crit = _mm_cmp_ps(__x, _mm_setzero_ps(), _CMP_LT_OS);
    dst = _mm_blendv_ps(dst, _mm_sub_ps(_mm_set1_ps(3.14159274f), dst), _crit);
    _crit = _mm_cmp_ps(__y, _mm_setzero_ps(), _CMP_LT_OS);
    dst = decx::utils::simd::_mm_signinv_ps_masked(dst, _crit);

    return dst;
}


inline __m128 decx::utils::simd::_mm_cos_ps(const __m128 __x)
{
    const __m128i _full_period_num = _mm_cvtps_epi32(_mm_div_ps(__x, _mm_set1_ps(3.1415926f)));
    // The normalized input Xs [-Pi, Pi]
    const __m128 _normed = _mm_sub_ps(__x, _mm_mul_ps(_mm_cvtepi32_ps(_full_period_num), _mm_set1_ps(3.1415926f)));
    
    const __m128 _x_sqrt = _mm_mul_ps(_normed, _normed);
    __m128 _x_term = _mm_div_ps(_x_sqrt, _mm_set1_ps(2));
    __m128 _res = _mm_sub_ps(_mm_set1_ps(1), _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(12)));
    _res = _mm_add_ps(_res, _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(30)));
    _res = _mm_sub_ps(_res, _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(56)));
    _res = _mm_add_ps(_res, _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(90)));
    _res = _mm_sub_ps(_res, _x_term);

    // Odd or even
    __m128i _mask_shfl = _mm_slli_epi32(_mm_and_si128(_full_period_num, _mm_set1_epi32(0x01)), 31);
    return _mm_xor_ps(_res, _mm_castsi128_ps(_mask_shfl));
}




inline __m128 decx::utils::simd::_mm_sin_ps(const __m128 __x)
{
    const __m128 _shitfed = _mm_sub_ps(__x, _mm_set1_ps(1.5707963));
    const __m128i _full_period_num = _mm_cvtps_epi32(_mm_div_ps(_shitfed, _mm_set1_ps(3.1415926f)));
    // The normalized input Xs [-Pi, Pi]
    const __m128 _normed = _mm_sub_ps(_shitfed, _mm_mul_ps(_mm_cvtepi32_ps(_full_period_num), _mm_set1_ps(3.1415926f)));
    
    const __m128 _x_sqrt = _mm_mul_ps(_normed, _normed);
    __m128 _x_term = _mm_div_ps(_x_sqrt, _mm_set1_ps(2));
    __m128 _res = _mm_sub_ps(_mm_set1_ps(1), _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(12)));
    _res = _mm_add_ps(_res, _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(30)));
    _res = _mm_sub_ps(_res, _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(56)));
    _res = _mm_add_ps(_res, _x_term);

    _x_term = _mm_mul_ps(_x_term, _mm_div_ps(_x_sqrt, _mm_set1_ps(90)));
    _res = _mm_sub_ps(_res, _x_term);

    // Odd or even
    __m128i _mask_shfl = _mm_slli_epi32(_mm_and_si128(_full_period_num, _mm_set1_epi32(0x01)), 31);
    return _mm_xor_ps(_res, _mm_castsi128_ps(_mask_shfl));
}

#endif      // #ifdef _DECX_CPU_PARTS_

#endif
