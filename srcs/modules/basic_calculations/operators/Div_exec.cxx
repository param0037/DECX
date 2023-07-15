/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Div_exec.h"



// ----------------------------------------- callers -----------------------------------------------------------

_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_m_fvec8_ST(const float* __restrict A, const float* __restrict B, float* __restrict dst, size_t len)
{
    __m256 tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i){
        tmpA = _mm256_load_ps(A + (i << 3));
        tmpB = _mm256_load_ps(B + (i << 3));

        tmpdst = _mm256_div_ps(tmpA, tmpB);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_m_ivec8_ST(const int* __restrict A, const int* __restrict B, int* __restrict dst, size_t len)
{
    __m256i tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_castps_si256(_mm256_load_ps((float*)(A + (i << 3))));
        tmpB = _mm256_castps_si256(_mm256_load_ps((float*)(B + (i << 3))));

#ifdef Windows
        tmpdst = _mm256_div_epi32(tmpA, tmpB);
#endif

#ifdef Linux
        ((int*)&tmpA)[0] /= ((const float*)&tmpB)[0];
        ((int*)&tmpA)[1] /= ((const float*)&tmpB)[1];
        ((int*)&tmpA)[2] /= ((const float*)&tmpB)[2];
        ((int*)&tmpA)[3] /= ((const float*)&tmpB)[3];
        ((int*)&tmpA)[4] /= ((const float*)&tmpB)[4];
        ((int*)&tmpA)[5] /= ((const float*)&tmpB)[5];
        ((int*)&tmpA)[6] /= ((const float*)&tmpB)[6];
        ((int*)&tmpA)[7] /= ((const float*)&tmpB)[7];
#endif

        _mm256_store_ps((float*)(dst + (i << 3)), _mm256_castsi256_ps(tmpdst));
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_m_dvec4_ST(const double* __restrict A, const double* __restrict B, double* __restrict dst, size_t len)
{
    __m256d tmpA, tmpB, tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpA = _mm256_load_pd(A + (i << 2));
        tmpB = _mm256_load_pd(B + (i << 2));

        tmpdst = _mm256_div_pd(tmpA, tmpB);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}



_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_c_fvec8_ST(const float* __restrict src, const float __x, float* __restrict dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_ps(src + (i << 3));

        tmpdst = _mm256_div_ps(tmpsrc, tmpX);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_c_ivec8_ST(const int* __restrict src, const int __x, int* __restrict dst, size_t len)
{
    __m256i tmpsrc, tmpX = _mm256_set1_epi32(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_castps_si256(_mm256_load_ps((float*)(src + (i << 3))));

#ifdef Windows
        tmpdst = _mm256_div_epi32(tmpsrc, tmpX);
#endif

#ifdef Linux
        ((int*)&tmpsrc)[0] /= ((const float*)&tmpX)[0];
        ((int*)&tmpsrc)[1] /= ((const float*)&tmpX)[1];
        ((int*)&tmpsrc)[2] /= ((const float*)&tmpX)[2];
        ((int*)&tmpsrc)[3] /= ((const float*)&tmpX)[3];
        ((int*)&tmpsrc)[4] /= ((const float*)&tmpX)[4];
        ((int*)&tmpsrc)[5] /= ((const float*)&tmpX)[5];
        ((int*)&tmpsrc)[6] /= ((const float*)&tmpX)[6];
        ((int*)&tmpsrc)[7] /= ((const float*)&tmpX)[7];
#endif

        _mm256_store_ps((float*)(dst + (i << 3)), _mm256_castsi256_ps(tmpdst));
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_c_dvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmpX = _mm256_set1_pd(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + (i << 2));

        tmpdst = _mm256_div_pd(tmpsrc, tmpX);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_cinv_fvec8_ST(const float* __restrict src, const float __x, float* __restrict dst, size_t len)
{
    __m256 tmpsrc, tmpX = _mm256_set1_ps(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_ps(src + (i << 3));

        tmpdst = _mm256_div_ps(tmpX, tmpsrc);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_cinv_ivec8_ST(const int* __restrict src, const int __x, int* __restrict dst, size_t len)
{
    __m256i tmpsrc, tmpX = _mm256_set1_epi32(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_castps_si256(_mm256_load_ps((float*)(src + i)));
#ifdef Windows
        tmpdst = _mm256_div_epi32(tmpX, tmpsrc);
#endif

#ifdef Linux
        ((int*)&tmpdst)[0] /= ((const float*)&tmpX)[0];
        ((int*)&tmpdst)[1] /= ((const float*)&tmpX)[1];
        ((int*)&tmpdst)[2] /= ((const float*)&tmpX)[2];
        ((int*)&tmpdst)[3] /= ((const float*)&tmpX)[3];
        ((int*)&tmpdst)[4] /= ((const float*)&tmpX)[4];
        ((int*)&tmpdst)[5] /= ((const float*)&tmpX)[5];
        ((int*)&tmpdst)[6] /= ((const float*)&tmpX)[6];
        ((int*)&tmpdst)[7] /= ((const float*)&tmpX)[7];
#endif
        _mm256_store_ps((float*)(dst + (i << 3)), _mm256_castsi256_ps(tmpdst));
    }
}


_THREAD_FUNCTION_ void 
decx::calc::CPUK::div_cinv_dvec4_ST(const double* __restrict src, const double __x, double* __restrict dst, size_t len)
{
    __m256d tmpsrc, tmpX = _mm256_set1_pd(__x), tmpdst;
    for (uint i = 0; i < len; ++i) {
        tmpsrc = _mm256_load_pd(src + (i << 2));

        tmpdst = _mm256_div_pd(tmpX, tmpsrc);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}