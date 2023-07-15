/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "clip_range.h"


_THREAD_FUNCTION_ void 
decx::calc::CPUK::_clip_range_fp32(const float* src, const float2 range, float* dst, const uint64_t proc_len)
{
    __m256 recv, tmpdst, tmp_crit;
    const __m256 lower_bound_v8 = _mm256_set1_ps(range.x);
    const __m256 upper_bound_v8 = _mm256_set1_ps(range.y);
    for (uint i = 0; i < proc_len; ++i) {
        recv = _mm256_load_ps(src + (i << 3));
        // clip at the lower bound
        tmp_crit = _mm256_cmp_ps(recv, lower_bound_v8, _CMP_GE_OS);
        tmpdst = _mm256_and_ps(recv, tmp_crit);
        // clip at the upper bound
        tmp_crit = _mm256_cmp_ps(recv, upper_bound_v8, _CMP_LE_OS);
        tmpdst = _mm256_and_ps(tmpdst, tmp_crit);

        _mm256_store_ps(dst + (i << 3), tmpdst);
    }
}



_THREAD_FUNCTION_ void
decx::calc::CPUK::_clip_range_fp64(const double* src, const double2 range, double* dst, const uint64_t proc_len)
{
    __m256d recv, tmpdst, tmp_crit;
    const __m256d lower_bound_v8 = _mm256_set1_pd(range.x);
    const __m256d upper_bound_v8 = _mm256_set1_pd(range.y);
    for (uint i = 0; i < proc_len; ++i) {
        recv = _mm256_load_pd(src + (i << 2));
        // clip at the lower bound
        tmp_crit = _mm256_cmp_pd(recv, lower_bound_v8, _CMP_GE_OS);
        tmpdst = _mm256_and_pd(recv, tmp_crit);
        // clip at the upper bound
        tmp_crit = _mm256_cmp_pd(recv, upper_bound_v8, _CMP_LE_OS);
        tmpdst = _mm256_and_pd(tmpdst, tmp_crit);

        _mm256_store_pd(dst + (i << 2), tmpdst);
    }
}