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


#ifndef _SIMD_FAST_MATH_AVX2_H_
#define _SIMD_FAST_MATH_AVX2_H_

#ifdef _DECX_CPU_PARTS_


#include "../../decx_utils_macros.h"
#include "../../vector_defines.h"


extern "C" _DECX_API_ __m128 __VECTORCALL__ _avx_cos_fp32x4(__m128);
extern "C" _DECX_API_ __m128 __VECTORCALL__ _avx_sin_fp32x4(__m128);
extern "C" _DECX_API_ __m128d __VECTORCALL__ _avx_cos_fp64x2(__m128d);
extern "C" _DECX_API_ __m128d __VECTORCALL__ _avx_sin_fp64x2(__m128d);
extern "C" _DECX_API_ __m256 __VECTORCALL__ _avx_cos_fp32x8(__m256);
extern "C" _DECX_API_ __m256 __VECTORCALL__ _avx_sin_fp32x8(__m256);
extern "C" _DECX_API_ __m256d __VECTORCALL__ _avx_cos_fp64x4(__m256d);
extern "C" _DECX_API_ __m256d __VECTORCALL__ _avx_sin_fp64x4(__m256d);


namespace decx
{
namespace utils{
    namespace simd
    {
        FORCEINLINE __m256 _mm256_abs_ps(__m256 __proc);
        FORCEINLINE __m256 _mm256_signinv_ps(__m256 __proc);
        FORCEINLINE __m256 _mm256_signinv_ps_masked(__m256 __proc, __m256 __mask);


        FORCEINLINE __m256d _mm256_abs_pd(__m256d __proc);
        FORCEINLINE __m256d _mm256_signinv_pd(__m256d __proc);

        inline __m256 _mm256_atan2_ps(const __m256 __y, const __m256 __x);
        inline __m256 _mm256_atan_ps(const __m256 __x);
        inline __m256 _mm256_cos_ps(const __m256 __x);
        inline __m256 _mm256_sin_ps(const __m256 __x);
    }
}
}



FORCEINLINE __m256 decx::utils::simd::_mm256_abs_ps(__m256 __proc)
{
    return _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(__proc), _mm256_set1_epi32(0x7fffffff)));
}


FORCEINLINE __m256 decx::utils::simd::_mm256_signinv_ps(__m256 __proc)
{
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(__proc), _mm256_set1_epi32(0x80000000)));
}


FORCEINLINE __m256d decx::utils::simd::_mm256_abs_pd(__m256d __proc)
{
    return _mm256_castsi256_pd(_mm256_and_si256(_mm256_castpd_si256(__proc), _mm256_set1_epi64x(0x7fffffffffffffff)));
}


FORCEINLINE __m256d decx::utils::simd::_mm256_signinv_pd(__m256d __proc)
{
    return _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(__proc), _mm256_set1_epi64x(0x8000000000000000)));
}


FORCEINLINE __m256 decx::utils::simd::_mm256_signinv_ps_masked(__m256 __proc, __m256 __mask)
{
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(__proc),
        _mm256_and_si256(_mm256_castps_si256(__mask), _mm256_set1_epi32(0x80000000))));
}


inline __m256 decx::utils::simd::_mm256_atan2_ps(const __m256 __y, const __m256 __x)
{
    __m256 dst;

    const __m256 _ax = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(__x), _mm256_set1_epi32(0x7FFFFFFF)));
    const __m256 _ay = _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(__y), _mm256_set1_epi32(0x7FFFFFFF)));
    const __m256 _a = _mm256_div_ps(_mm256_min_ps(_ax, _ay), _mm256_max_ps(_ax, _ay));
    const __m256 _s = _mm256_mul_ps(_a, _a);
    const __m256 _r1 = _mm256_fmadd_ps(_s, _mm256_set1_ps(-0.0464964749), _mm256_set1_ps(0.15931422));
    const __m256 _r2 = _mm256_fmadd_ps(_r1, _s, _mm256_set1_ps(-0.327622764));
    const __m256 _r3 = _mm256_mul_ps(_a, _mm256_fmadd_ps(_r2, _s, _mm256_set1_ps(1)));
    dst = _r3;
    __m256 _crit = _mm256_cmp_ps(_ay, _ax, _CMP_GT_OS);
    dst = _mm256_blendv_ps(dst, _mm256_sub_ps(_mm256_set1_ps(1.57079637f), dst), _crit);
    _crit = _mm256_cmp_ps(__x, _mm256_setzero_ps(), _CMP_LT_OS);
    dst = _mm256_blendv_ps(dst, _mm256_sub_ps(_mm256_set1_ps(3.14159274f), dst), _crit);
    _crit = _mm256_and_ps(__y, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    return _mm256_xor_ps(dst, _crit);
}




inline __m256 decx::utils::simd::_mm256_atan_ps(__m256 __x)
{
    using namespace decx::utils::simd;
    const __m256 _abs_GT1 = _mm256_cmp_ps(_mm256_abs_ps(__x), _mm256_set1_ps(1.f), _CMP_GT_OS);
    const __m256 _neg = _mm256_cmp_ps(__x, _mm256_set1_ps(0.f), _CMP_LE_OS);
    const __m256 _norm = _mm256_blendv_ps(_mm256_abs_ps(__x),
        _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_abs_ps(__x)), _abs_GT1);

    const __m256 _s = _mm256_mul_ps(_norm, _norm);
    const __m256 _r1 = _mm256_fmadd_ps(_s, _mm256_set1_ps(-0.0464964749), _mm256_set1_ps(0.15931422));
    const __m256 _r2 = _mm256_fmadd_ps(_r1, _s, _mm256_set1_ps(-0.327622764));
    const __m256 _r3 = _mm256_mul_ps(_norm, _mm256_fmadd_ps(_r2, _s, _mm256_set1_ps(1)));

    __m256 dst1 = _mm256_blendv_ps(_r3, _mm256_sub_ps(_mm256_set1_ps(1.57079637f), _r3), _abs_GT1);
    return _mm256_signinv_ps_masked(dst1, _neg);
}




inline __m256 decx::utils::simd::_mm256_cos_ps(const __m256 __x)
{
    const __m256i _full_period_num = _mm256_cvtps_epi32(_mm256_div_ps(__x, _mm256_set1_ps(3.1415926f)));
    // The normalized input Xs [-Pi_FP32, Pi_FP32]
    const __m256 _normed = _mm256_sub_ps(__x, _mm256_mul_ps(_mm256_cvtepi32_ps(_full_period_num), _mm256_set1_ps(3.1415926f)));
    
    const __m256 _x_sqrt = _mm256_mul_ps(_normed, _normed);
    __m256 _x_term = _mm256_div_ps(_x_sqrt, _mm256_set1_ps(2));
    __m256 _res = _mm256_sub_ps(_mm256_set1_ps(1), _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(12)));
    _res = _mm256_add_ps(_res, _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(30)));
    _res = _mm256_sub_ps(_res, _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(56)));
    _res = _mm256_add_ps(_res, _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(90)));
    _res = _mm256_sub_ps(_res, _x_term);

    // Odd or even
    __m256i _mask_shfl = _mm256_slli_epi32(_mm256_and_si256(_full_period_num, _mm256_set1_epi32(0x01)), 31);
    return _mm256_xor_ps(_res, _mm256_castsi256_ps(_mask_shfl));
}




inline __m256 decx::utils::simd::_mm256_sin_ps(const __m256 __x)
{
    const __m256 _shitfed = _mm256_sub_ps(__x, _mm256_set1_ps(1.5707963));
    const __m256i _full_period_num = _mm256_cvtps_epi32(_mm256_div_ps(_shitfed, _mm256_set1_ps(3.1415926f)));
    // The normalized input Xs [-Pi_FP32, Pi_FP32]
    const __m256 _normed = _mm256_sub_ps(_shitfed, _mm256_mul_ps(_mm256_cvtepi32_ps(_full_period_num), _mm256_set1_ps(3.1415926f)));
    
    const __m256 _x_sqrt = _mm256_mul_ps(_normed, _normed);
    __m256 _x_term = _mm256_div_ps(_x_sqrt, _mm256_set1_ps(2));
    __m256 _res = _mm256_sub_ps(_mm256_set1_ps(1), _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(12)));
    _res = _mm256_add_ps(_res, _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(30)));
    _res = _mm256_sub_ps(_res, _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(56)));
    _res = _mm256_add_ps(_res, _x_term);

    _x_term = _mm256_mul_ps(_x_term, _mm256_div_ps(_x_sqrt, _mm256_set1_ps(90)));
    _res = _mm256_sub_ps(_res, _x_term);

    // Odd or even
    __m256i _mask_shfl = _mm256_slli_epi32(_mm256_and_si256(_full_period_num, _mm256_set1_epi32(0x01)), 31);
    return _mm256_xor_ps(_res, _mm256_castsi256_ps(_mask_shfl));
}

#endif      // #ifdef _DECX_CPU_PARTS_

#endif