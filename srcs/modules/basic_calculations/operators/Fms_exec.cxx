/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Fms_exec.h"



_THREAD_FUNCTION_ void 
decx::calc::CPUK::fms_m_fvec8_ST(const float* __restrict A, const float* __restrict B, const float* __restrict C, float* __restrict dst, size_t len)
{
    __m256 tmpA, tmpB, tmpC, tmpdst;

    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_ps(A + (i << 3));
        tmpB = _mm256_load_ps(B + (i << 3));
        tmpC = _mm256_load_ps(C + (i << 3));

        tmpdst = _mm256_fmsub_ps(tmpA, tmpB, tmpC);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::fms_m_ivec8_ST(const int* __restrict A, const int* __restrict B, const int* __restrict C, int* __restrict dst, size_t len)
{
    __m256i tmpA, tmpB, tmpC, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_castps_si256(_mm256_load_ps((float*)(A + (i << 3))));
        tmpB = _mm256_castps_si256(_mm256_load_ps((float*)(B + (i << 3))));
        tmpC = _mm256_castps_si256(_mm256_load_ps((float*)(C + (i << 3))));

        tmpdst = _mm256_mullo_epi32(tmpA, tmpB);
        tmpdst = _mm256_sub_epi32(tmpdst, tmpC);

        _mm256_store_ps((float*)(dst + (i << 3)), _mm256_castsi256_ps(tmpdst));
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::fms_m_dvec4_ST(const double* __restrict A, const double* __restrict B, const double* __restrict C, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpB, tmpC, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + (i << 2));
        tmpB = _mm256_load_pd(B + (i << 2));
        tmpC = _mm256_load_pd(C + (i << 2));

        tmpdst = _mm256_fmsub_pd(tmpA, tmpB, tmpC);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}


// dst = A * __x - B
_THREAD_FUNCTION_ void 
decx::calc::CPUK::fms_c_fvec8_ST(const float* __restrict A, const float __x, const float* __restrict B, float* __restrict dst, size_t len)
{
    __m256 tmpA, tmpX = _mm256_set1_ps(__x), tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_ps(A + (i << 3));
        tmpB = _mm256_load_ps(B + (i << 3));

        tmpdst = _mm256_fmsub_ps(tmpA, tmpX, tmpB);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


// dst = A * __x1 - __x2
_THREAD_FUNCTION_ void
decx::calc::CPUK::fms_c2_fvec8_ST(const float* __restrict A, const float __x1, const float __x2, float* __restrict dst, size_t len)
{
    __m256 tmpA, tmpdst;
    const __m256 tmpX1 = _mm256_set1_ps(__x1),tmpX2 = _mm256_set1_ps(__x2);
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_ps(A + (i << 3));

        tmpdst = _mm256_fmsub_ps(tmpA, tmpX1, tmpX2);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::fms_c_ivec8_ST(const int* __restrict A, const int __x, const int* __restrict B, int* __restrict dst, size_t len)
{
    __m256i tmpA, tmpB, tmpdst;
    const __m256i _constant = _mm256_set1_epi32(__x);
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_castps_si256(_mm256_load_ps((float*)(A + (i << 3))));
        tmpB = _mm256_castps_si256(_mm256_load_ps((float*)(B + (i << 3))));

        tmpdst = _mm256_mullo_epi32(tmpA, _constant);
        tmpdst = _mm256_sub_epi32(tmpdst, tmpB);

        _mm256_store_ps((float*)(dst + (i << 3)), _mm256_castsi256_ps(tmpdst));
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::fms_c_dvec4_ST(const double* __restrict A, const double __x, const double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpX = _mm256_set1_pd(__x), tmpdst, tmpB;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + (i << 2));
        tmpB = _mm256_load_pd(B + (i << 2));

        tmpdst = _mm256_fmsub_pd(tmpA, tmpX, tmpB);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}