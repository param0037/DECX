/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "cmp_exec.h"


// fp32
_THREAD_FUNCTION_ void
decx::bp::CPUK::_maximum_vec8_fp32_1D(const float* __restrict src, const uint64_t len, float* res_vec,
    const uint8_t _occupied_length)
{
    __m256 tmp_recv, sum_vec8 = _mm256_load_ps(src);
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v8(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm256_load_ps(src + ((uint64_t)i << 3));
        sum_vec8 = _mm256_max_ps(tmp_recv, sum_vec8);
    }
    tmp_recv = _mm256_load_ps(src + ((uint64_t)(len - 1) << 3));
    tmp_recv = _mm256_permutevar8x32_ps(tmp_recv, _shuffle_var);
    sum_vec8 = _mm256_max_ps(tmp_recv, sum_vec8);

    float res = decx::utils::simd::_mm256_h_max(sum_vec8);
    *res_vec = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_min_max_vec8_fp32_1D(const float* __restrict src, const uint64_t len, float* res_min,
    float* res_max, const uint8_t _occupied_length)
{
    __m256 tmp_recv, max_vec8 = _mm256_load_ps(src);
    __m256 min_vec8 = max_vec8;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v8(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm256_load_ps(src + ((uint64_t)i << 3));
        max_vec8 = _mm256_max_ps(tmp_recv, max_vec8);
        min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);
    }
    tmp_recv = _mm256_load_ps(src + ((uint64_t)(len - 1) << 3));
    tmp_recv = _mm256_permutevar8x32_ps(tmp_recv, _shuffle_var);
    max_vec8 = _mm256_max_ps(tmp_recv, max_vec8);
    min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);

    float res = decx::utils::simd::_mm256_h_min(min_vec8);
    *res_min = res;
    res = decx::utils::simd::_mm256_h_max(max_vec8);
    *res_max = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_maximum_vec8_fp32_2D(const float* __restrict src, const uint2 _proc_dims, float* res_vec,
    const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m256 tmp_recv, sum_vec8 = _mm256_load_ps(src);
    uint64_t dex_src = 0;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v8(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm256_load_ps(src + dex_src);
            sum_vec8 = _mm256_max_ps(tmp_recv, sum_vec8);
            dex_src += 8;
        }
        tmp_recv = _mm256_load_ps(src + dex_src);
        tmp_recv = _mm256_permutevar8x32_ps(tmp_recv, _shuffle_var);
        sum_vec8 = _mm256_max_ps(tmp_recv, sum_vec8);
    }

    float res = decx::utils::simd::_mm256_h_max(sum_vec8);
    *res_vec = res;
}




_THREAD_FUNCTION_ void
decx::bp::CPUK::_min_max_vec8_fp32_2D(const float* __restrict src, const uint2 _proc_dims, float* res_min,
    float* res_max, const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m256 tmp_recv, min_vec8 = _mm256_load_ps(src);
    __m256 max_vec8 = min_vec8;
    uint64_t dex_src = 0;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v8(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm256_load_ps(src + dex_src);
            min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);
            max_vec8 = _mm256_max_ps(tmp_recv, max_vec8);
            dex_src += 8;
        }
        tmp_recv = _mm256_load_ps(src + dex_src);
        tmp_recv = _mm256_permutevar8x32_ps(tmp_recv, _shuffle_var);
        min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);
        max_vec8 = _mm256_max_ps(tmp_recv, max_vec8);
    }

    float res = decx::utils::simd::_mm256_h_min(min_vec8);
    *res_min = res;
    res = decx::utils::simd::_mm256_h_max(max_vec8);
    *res_max = res;
}


// uint8
_THREAD_FUNCTION_ void
decx::bp::CPUK::_maximum_vec16_uint8_1D(const uint8_t* __restrict src, const uint64_t len, uint8_t* res_vec,
    const uint8_t _occupied_length)
{
    __m128i tmp_recv, sum_vec8 = _mm_castps_si128(_mm_load_ps((float*)src));
    const __m128i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v16(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + ((uint64_t)i << 4))));
        sum_vec8 = _mm_max_epu8(tmp_recv, sum_vec8);
    }
    tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + ((uint64_t)(len - 1) << 4))));
    tmp_recv = _mm_shuffle_epi8(tmp_recv, _shuffle_var);
    sum_vec8 = _mm_max_epu8(tmp_recv, sum_vec8);

    uint8_t res = decx::utils::simd::_mm128_h_max_u8(sum_vec8);
    *res_vec = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_min_max_vec16_uint8_1D(const uint8_t* __restrict src, const uint64_t len, uint8_t* res_min,
    uint8_t* res_max, const uint8_t _occupied_length)
{
    __m128i tmp_recv, min_vec8 = _mm_castps_si128(_mm_load_ps((float*)src));
    __m128i max_vec8 = min_vec8;
    const __m128i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v16(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + ((uint64_t)i << 4))));
        min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
        max_vec8 = _mm_max_epu8(tmp_recv, max_vec8);
    }
    tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + ((uint64_t)(len - 1) << 4))));
    tmp_recv = _mm_shuffle_epi8(tmp_recv, _shuffle_var);
    min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
    max_vec8 = _mm_max_epu8(tmp_recv, max_vec8);

    uint8_t res = decx::utils::simd::_mm128_h_min_u8(min_vec8);
    *res_min = res;
    res = decx::utils::simd::_mm128_h_max_u8(max_vec8);
    *res_max = res;
}


_THREAD_FUNCTION_ void
decx::bp::CPUK::_maximum_vec16_uint8_2D(const uint8_t* __restrict src, const uint2 _proc_dims, uint8_t* res_vec,
    const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m128i tmp_recv, sum_vec8 = _mm_castps_si128(_mm_load_ps((float*)src));
    uint64_t dex_src = 0;
    const __m128i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v16(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + dex_src)));
            sum_vec8 = _mm_max_epu8(tmp_recv, sum_vec8);
            dex_src += 16;
        }
        tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + dex_src)));
        tmp_recv = _mm_shuffle_epi8(tmp_recv, _shuffle_var);
        sum_vec8 = _mm_max_epu8(tmp_recv, sum_vec8);
    }

    uint8_t res = decx::utils::simd::_mm128_h_max_u8(sum_vec8);
    *res_vec = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_min_max_vec16_uint8_2D(const uint8_t* __restrict src, const uint2 _proc_dims, uint8_t* res_min,
    uint8_t* res_max, const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m128i tmp_recv, min_vec8 = _mm_castps_si128(_mm_load_ps((float*)src));
    __m128i max_vec8 = min_vec8;
    uint64_t dex_src = 0;
    const __m128i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v16(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + dex_src)));
            min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
            max_vec8 = _mm_max_epu8(tmp_recv, max_vec8);
            dex_src += 16;
        }
        tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + dex_src)));
        tmp_recv = _mm_shuffle_epi8(tmp_recv, _shuffle_var);
        min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
        max_vec8 = _mm_max_epu8(tmp_recv, max_vec8);
    }

    uint8_t res = decx::utils::simd::_mm128_h_min_u8(min_vec8);
    *res_min = res;
    res = decx::utils::simd::_mm128_h_max_u8(max_vec8);
    *res_max = res;
}


// double
_THREAD_FUNCTION_ void
decx::bp::CPUK::_maximum_vec4_fp64_2D(const double* __restrict src, const uint2 _proc_dims, double* res_vec,
    const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m256d tmp_recv, max_vec4 = _mm256_load_pd(src);
    uint64_t dex_src = 0;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v4(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm256_load_pd(src + dex_src);
            max_vec4 = _mm256_max_pd(tmp_recv, max_vec4);
            dex_src += 4;
        }
        tmp_recv = _mm256_load_pd(src + dex_src);
        tmp_recv = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(tmp_recv), _shuffle_var));
        max_vec4 = _mm256_max_pd(tmp_recv, max_vec4);
    }

    double res = decx::utils::simd::_mm256d_h_max(max_vec4);
    *res_vec = res;
}


_THREAD_FUNCTION_ void
decx::bp::CPUK::_min_max_vec4_fp64_2D(const double* __restrict src, const uint2 _proc_dims, double* res_min,
    double* res_max, const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m256d tmp_recv, min_vec4 = _mm256_load_pd(src);
    __m256d max_vec4 = min_vec4;
    uint64_t dex_src = 0;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v4(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm256_load_pd(src + dex_src);
            min_vec4 = _mm256_min_pd(tmp_recv, min_vec4);
            max_vec4 = _mm256_max_pd(tmp_recv, max_vec4);
            dex_src += 4;
        }
        tmp_recv = _mm256_load_pd(src + dex_src);
        tmp_recv = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(tmp_recv), _shuffle_var));
        min_vec4 = _mm256_min_pd(tmp_recv, min_vec4);
        max_vec4 = _mm256_max_pd(tmp_recv, max_vec4);
    }

    double res = decx::utils::simd::_mm256d_h_min(min_vec4);
    *res_min = res;
    res = decx::utils::simd::_mm256d_h_max(max_vec4);
    *res_max = res;
}


_THREAD_FUNCTION_ void
decx::bp::CPUK::_maximum_vec4_fp64_1D(const double* __restrict src, const size_t len, double* res_vec,
    const uint8_t _occupied_length)
{
    __m256d tmp_recv, sum_vec4 = _mm256_load_pd(src);
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v4(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm256_load_pd(src + ((size_t)i << 2));
        sum_vec4 = _mm256_max_pd(tmp_recv, sum_vec4);
    }
    tmp_recv = _mm256_load_pd(src + ((uint64_t)(len - 1) << 2));
    tmp_recv = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(tmp_recv), _shuffle_var));
    sum_vec4 = _mm256_max_pd(tmp_recv, sum_vec4);

    double res = decx::utils::simd::_mm256d_h_max(sum_vec4);
    *res_vec = res;
}


_THREAD_FUNCTION_ void
decx::bp::CPUK::_min_max_vec4_fp64_1D(const double* __restrict src, const size_t len, double* res_min,
    double* res_max, const uint8_t _occupied_length)
{
    __m256d tmp_recv, min_vec4 = _mm256_load_pd(src);
    __m256d max_vec4 = min_vec4;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v4(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm256_load_pd(src + ((size_t)i << 2));
        min_vec4 = _mm256_min_pd(tmp_recv, min_vec4);
        max_vec4 = _mm256_max_pd(tmp_recv, max_vec4);
    }
    tmp_recv = _mm256_load_pd(src + ((uint64_t)(len - 1) << 2));
    tmp_recv = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(tmp_recv), _shuffle_var));
    min_vec4 = _mm256_min_pd(tmp_recv, min_vec4);
    max_vec4 = _mm256_max_pd(tmp_recv, max_vec4);

    double res = decx::utils::simd::_mm256d_h_min(min_vec4);
    *res_min = res;
    res = decx::utils::simd::_mm256d_h_max(max_vec4);
    *res_max = res;
}