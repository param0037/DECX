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


#ifndef _SIMD_FAST_MATH_SSE_H_
#define _SIMD_FAST_MATH_SSE_H_

#ifdef _DECX_CPU_PARTS_

#include "decx_utils_macros.h"
#include "../vector_defines.h"

/*
* DO NOT use when requiring very exact precision ! For example,
* large signal FFT (N >= 1024)
*/
//extern "C" __m128 __vectorcall fast_mm_cos_ps(__m128 __x);
//extern "C" __m128 __vectorcall fast_mm_sin_ps(__m128 __x);


extern "C" __m128 __vectorcall _avx_cos_fp32x4(__m128);
extern "C" __m128 __vectorcall _avx_sin_fp32x4(__m128);


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
        //inline __m128 _mm_cos_ps(const __m128 __x);
        //inline __m128 _mm_sin_ps(const __m128 __x);
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

//
//__m128 decx::utils::simd::_mm_cos_ps(__m128 __x)
//{
//    return fast_mm_cos_ps(__x);
//}
//
//
//__m128 decx::utils::simd::_mm_sin_ps(__m128 __x)
//{
//    return fast_mm_sin_ps(__x);
//}

#endif      // #ifdef _DECX_CPU_PARTS_

#endif
