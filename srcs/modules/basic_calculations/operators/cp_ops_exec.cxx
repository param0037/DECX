/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "cp_ops_exec.h"




_THREAD_FUNCTION_ void 
decx::calc::CPUK::cp_add_m_fvec4_ST(const double* __restrict A, const double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpB;
    __m256 tmpdst;
    for (uint i = 0; i < len; ++i){
        tmpA = _mm256_load_pd(A + ((size_t)i << 2));
        tmpB = _mm256_load_pd(B + ((size_t)i << 2));

        tmpdst = _mm256_add_ps(_mm256_castpd_ps(tmpA), _mm256_castpd_ps(tmpB));

        _mm256_store_pd(dst + ((size_t)i << 2), _mm256_castps_pd(tmpdst));
    }
}



_THREAD_FUNCTION_ void 
decx::calc::CPUK::cp_sub_m_fvec4_ST(const double* __restrict A, const double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpB;
    __m256 tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + ((size_t)i << 2));
        tmpB = _mm256_load_pd(B + ((size_t)i << 2));

        tmpdst = _mm256_sub_ps(_mm256_castpd_ps(tmpA), _mm256_castpd_ps(tmpB));

        _mm256_store_pd(dst + ((size_t)i << 2), _mm256_castps_pd(tmpdst));
    }
}



_THREAD_FUNCTION_ void 
decx::calc::CPUK::cp_mul_m_fvec4_ST(const double* __restrict A, const double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpB;
    __m256 tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + ((size_t)i << 2));
        tmpB = _mm256_load_pd(B + ((size_t)i << 2));

        tmpdst = decx::signal::CPUK::_cp4_mul_cp4_fp32(_mm256_castpd_ps(tmpA), _mm256_castpd_ps(tmpB));

        _mm256_store_pd(dst + ((size_t)i << 2), _mm256_castps_pd(tmpdst));
    }
}



_THREAD_FUNCTION_ void 
decx::calc::CPUK::cp_add_c_fvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmp_x = _mm256_set1_pd(__x);
    __m256 tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + ((size_t)i << 2));

        tmpdst = _mm256_add_ps(_mm256_castpd_ps(tmpsrc), _mm256_castpd_ps(tmp_x));

        _mm256_store_pd(dst + ((size_t)i << 2), _mm256_castps_pd(tmpdst));
    }
}



_THREAD_FUNCTION_ void 
decx::calc::CPUK::cp_sub_c_fvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmp_x = _mm256_set1_pd(__x);
    __m256 tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + ((size_t)i << 2));

        tmpdst = _mm256_sub_ps(_mm256_castpd_ps(tmpsrc), _mm256_castpd_ps(tmp_x));

        _mm256_store_pd(dst + ((size_t)i << 2), _mm256_castps_pd(tmpdst));
    }
}



_THREAD_FUNCTION_ void 
decx::calc::CPUK::cp_mul_c_fvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmp_x = _mm256_set1_pd(__x);
    __m256 tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + ((size_t)i << 2));

        tmpdst = decx::signal::CPUK::_cp4_mul_cp1_fp32(_mm256_castpd_ps(tmpsrc), *((de::CPf*)&__x));

        _mm256_store_pd(dst + ((size_t)i << 2), _mm256_castps_pd(tmpdst));
    }
}