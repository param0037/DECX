/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "Sub_exec.h"



// ----------------------------------------- callers -----------------------------------------------------------

_THREAD_FUNCTION_ void decx::calc::CPUK::sub_m_fvec8_ST(const float* __restrict A, float* __restrict B, float* __restrict dst, size_t len)
{
    __m256 tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_ps(A + (i << 3));
        tmpB = _mm256_load_ps(B + (i << 3));

        tmpdst = _mm256_sub_ps(tmpA, tmpB);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_m_ivec8_ST(const __m256i* __restrict A, __m256i* __restrict B, __m256i* __restrict dst, size_t len)
{
    __m256i tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_si256(A + i);
        tmpB = _mm256_load_si256(B + i);

        tmpdst = _mm256_sub_epi32(tmpA, tmpB);

        _mm256_store_si256(dst + i, tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_m_dvec4_ST(const double* __restrict A, double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + (i << 2));
        tmpB = _mm256_load_pd(B + (i << 2));

        tmpdst = _mm256_sub_pd(tmpA, tmpB);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}



_THREAD_FUNCTION_ void decx::calc::CPUK::sub_c_fvec8_ST(const float* __restrict src, const float __x, float* __restrict dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_ps(src + (i << 3));

        tmpdst = _mm256_sub_ps(tmpsrc, tmpX);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_c_ivec8_ST(const __m256i* __restrict src, const int __x, __m256i* __restrict dst, size_t len)
{
    __m256i tmpsrc, tmpX = _mm256_set1_epi32(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_si256(src + i);

        tmpdst = _mm256_sub_epi32(tmpsrc, tmpX);

        _mm256_store_si256(dst + i, tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_c_dvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmpX = _mm256_set1_pd(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + (i << 2));

        tmpdst = _mm256_sub_pd(tmpsrc, tmpX);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_cinv_fvec8_ST(const float* __restrict src, const float __x, float* __restrict dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_ps(src + (i << 3));

        tmpdst = _mm256_sub_ps(tmpX, tmpsrc);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_cinv_ivec8_ST(const __m256i* __restrict src, const int __x, __m256i* __restrict dst, size_t len)
{
    __m256i tmpsrc, tmpX = _mm256_set1_epi32(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_si256(src + i);

        tmpdst = _mm256_sub_epi32(tmpX, tmpsrc);

        _mm256_store_si256(dst + i, tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::sub_cinv_dvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmpX = _mm256_set1_pd(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + (i << 2));

        tmpdst = _mm256_sub_pd(tmpX, tmpsrc);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}

