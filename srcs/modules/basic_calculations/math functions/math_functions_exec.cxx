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


#include "math_functions_exec.h"


#define _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(__intrinsic) {      \
    __m256 tmpsrc, tmpdst;                                      \
    for (uint i = 0; i < len; ++i) {                            \
        tmpsrc = _mm256_load_ps(src + ((uint64_t)i << 3));      \
        tmpdst = __intrinsic(tmpsrc);                           \
        _mm256_store_ps(dst + ((uint64_t)i << 3), tmpdst);      \
    }                                                           \
}                                                               \


#define _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(__intrinsic) {      \
    __m256d tmpsrc, tmpdst;                                     \
    for (uint i = 0; i < len; ++i) {                            \
        tmpsrc = _mm256_load_pd(src + ((uint64_t)i << 2));      \
        tmpdst = __intrinsic(tmpsrc);                           \
        _mm256_store_pd(dst + ((uint64_t)i << 2), tmpdst);      \
    }                                                           \
}                                                               \



_THREAD_FUNCTION_ void
decx::calc::CPUK::log10_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_log10_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::log10_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len){
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_log10_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::log2_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_log2_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::log2_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_log2_pd);
#endif
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::sin_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_sin_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::sin_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_sin_pd);
#endif
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::cos_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_cos_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::cos_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_cos_pd);
#endif
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::tan_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_tan_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::tan_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_tan_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::exp_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_exp_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::exp_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_exp_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::acos_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_acos_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::acos_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_acos_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::asin_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_asin_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::asin_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_asin_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::atan_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_atan_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::atan_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_atan_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::sinh_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_sinh_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::sinh_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_sinh_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::cosh_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_cosh_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::cosh_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_cosh_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::tanh_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_tanh_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::tanh_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_tanh_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::abs_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(decx::utils::simd::_mm256_abs_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::abs_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(decx::utils::simd::_mm256_abs_pd);
#endif
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::sqrt_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_sqrt_ps);
#endif
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::sqrt_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
#ifdef _MSC_VER
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_sqrt_pd);
#endif
}