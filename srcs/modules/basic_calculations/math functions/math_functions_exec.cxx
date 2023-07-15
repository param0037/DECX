/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_log10_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::log10_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len){
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_log10_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::log2_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_log2_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::log2_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_log2_pd);
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::sin_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_sin_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::sin_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_sin_pd);
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::cos_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_cos_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::cos_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_cos_pd);
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::tan_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_tan_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::tan_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_tan_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::exp_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_exp_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::exp_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_exp_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::acos_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_acos_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::acos_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_acos_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::asin_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_asin_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::asin_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_asin_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::atan_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_atan_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::atan_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_atan_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::sinh_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_sinh_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::sinh_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_sinh_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::cosh_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_cosh_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::cosh_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_cosh_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::tanh_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_tanh_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::tanh_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_tanh_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::abs_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(decx::utils::simd::_mm256_abs_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::abs_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(decx::utils::simd::_mm256_abs_pd);
}


_THREAD_FUNCTION_ void
decx::calc::CPUK::sqrt_fvec8_ST(const float* __restrict src, float* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP32_(_mm256_sqrt_ps);
}
_THREAD_FUNCTION_ void
decx::calc::CPUK::sqrt_dvec4_ST(const double* __restrict src, double* __restrict dst, uint64_t len) {
    _MATH_SINGLE_INTRIN_AVX2_CALL_FP64_(_mm256_sqrt_pd);
}