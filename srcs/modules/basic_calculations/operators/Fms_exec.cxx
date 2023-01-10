/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "Fms_exec.h"



_THREAD_FUNCTION_ void decx::calc::CPUK::fms_m_fvec8_ST(const float* __restrict A, float* __restrict B, float* __restrict C, float* __restrict dst, size_t len)
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


_THREAD_FUNCTION_ void decx::calc::CPUK::fms_m_ivec8_ST(const __m256i* __restrict A, __m256i* __restrict B, __m256i* __restrict C, __m256i* __restrict dst, size_t len)
{
    __m256i tmpA, tmpB, tmpC, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_si256(A + i);
        tmpB = _mm256_load_si256(B + i);
        tmpC = _mm256_load_si256(C + i);
#ifdef Window
        tmpA.m256i_i32[0] *= tmpB.m256i_i32[0];
        tmpA.m256i_i32[1] *= tmpB.m256i_i32[1];
        tmpA.m256i_i32[2] *= tmpB.m256i_i32[2];
        tmpA.m256i_i32[3] *= tmpB.m256i_i32[3];
#endif

#ifdef Linux
        ((int*)&tmpA)[0] *= ((int*)&tmpB)[0];
        ((int*)&tmpA)[1] *= ((int*)&tmpB)[1];
        ((int*)&tmpA)[2] *= ((int*)&tmpB)[2];
        ((int*)&tmpA)[3] *= ((int*)&tmpB)[3];
#endif
        tmpdst = _mm256_sub_epi32(tmpA, tmpB);

        _mm256_store_si256(dst + i, tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::fms_m_dvec4_ST(const double* __restrict A, double* __restrict B, double* __restrict C, double* __restrict dst, size_t len)
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


// dst = A * __restrict __x - B
_THREAD_FUNCTION_ void decx::calc::CPUK::fms_c_fvec8_ST(const float* __restrict A, const float __x, float* __restrict B, float* __restrict dst, size_t len)
{
    __m256 tmpA, tmpX = _mm256_set1_ps(__x), tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_ps(A + (i << 3));
        tmpB = _mm256_load_ps(B + (i << 3));

        tmpdst = _mm256_fmsub_ps(tmpA, tmpX, tmpB);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::fms_c_ivec8_ST(const __m256i* __restrict A, const int __x, __m256i* __restrict B, __m256i* __restrict dst, size_t len)
{
    __m256i tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_si256(A + i);
        tmpB = _mm256_load_si256(B + i);
#ifdef Windows
        tmpA.m256i_i32[0] *= __x;
        tmpA.m256i_i32[1] *= __x;
        tmpA.m256i_i32[2] *= __x;
        tmpA.m256i_i32[3] *= __x;
#endif

#ifdef Linux
        ((int*)&tmpA)[0] *= __x;
        ((int*)&tmpA)[1] *= __x;
        ((int*)&tmpA)[2] *= __x;
        ((int*)&tmpA)[3] *= __x;
#endif

        tmpdst = _mm256_sub_epi32(tmpA, tmpB);

        _mm256_store_si256(dst + i, tmpdst);
    }
}


_THREAD_FUNCTION_ void decx::calc::CPUK::fms_c_dvec4_ST(const double* __restrict A, const double __x, double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpX = _mm256_set1_pd(__x), tmpdst, tmpB;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + (i << 2));
        tmpB = _mm256_load_pd(B + (i << 2));

        tmpdst = _mm256_fmsub_pd(tmpA, tmpX, tmpB);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}
