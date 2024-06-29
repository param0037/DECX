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


#include "cmp_exec.h"

// fp32

_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec8_fp32_1D(const float* __restrict src, const uint64_t len, float* res_vec,
    const uint8_t _occupied_length)
{
    __m256 tmp_recv, min_vec8 = _mm256_load_ps(src);
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v8(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm256_load_ps(src + ((uint64_t)i << 3));
        min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);
    }
    tmp_recv = _mm256_load_ps(src + ((uint64_t)(len - 1) << 3));
    tmp_recv = _mm256_permutevar8x32_ps(tmp_recv, _shuffle_var);
    min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);

    float res = decx::utils::simd::_mm256_h_min(min_vec8);
    *res_vec = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec8_fp32_2D(const float* __restrict src, const uint2 _proc_dims, float* res_vec,
    const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m256 tmp_recv, min_vec8 = _mm256_load_ps(src);
    uint64_t dex_src = 0;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v8(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm256_load_ps(src + dex_src);
            min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);
            dex_src += 8;
        }
        tmp_recv = _mm256_load_ps(src + dex_src);
        tmp_recv = _mm256_permutevar8x32_ps(tmp_recv, _shuffle_var);
        min_vec8 = _mm256_min_ps(tmp_recv, min_vec8);
    }

    float res = decx::utils::simd::_mm256_h_min(min_vec8);
    *res_vec = res;
}


// uint8
_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec16_uint8_1D(const uint8_t* __restrict src, const uint64_t len, uint8_t* res_vec,
    const uint8_t _occupied_length)
{
    __m128i tmp_recv, min_vec8 = _mm_castps_si128(_mm_load_ps((float*)src));
    const __m128i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v16(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + ((uint64_t)i << 4))));
        min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
    }
    tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + ((uint64_t)(len - 1) << 4))));
    tmp_recv = _mm_shuffle_epi8(tmp_recv, _shuffle_var);
    min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);

    uint8_t res = decx::utils::simd::_mm128_h_min_u8(min_vec8);
    *res_vec = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec16_uint8_2D(const uint8_t* __restrict src, const uint2 _proc_dims, uint8_t* res_vec,
    const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m128i tmp_recv, min_vec8 = _mm_castps_si128(_mm_load_ps((float*)src));
    uint64_t dex_src = 0;
    const __m128i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v16(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + dex_src)));
            min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
            dex_src += 16;
        }
        tmp_recv = _mm_castps_si128(_mm_load_ps((float*)(src + dex_src)));
        tmp_recv = _mm_shuffle_epi8(tmp_recv, _shuffle_var);
        min_vec8 = _mm_min_epu8(tmp_recv, min_vec8);
    }

    uint8_t res = decx::utils::simd::_mm128_h_min_u8(min_vec8);
    *res_vec = res;
}


// double

_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec4_fp64_2D(const double* __restrict src, const uint2 _proc_dims, double* res_vec,
    const uint32_t Wsrc, const uint8_t _occupied_length)
{
    __m256d tmp_recv, min_vec4 = _mm256_load_pd(src);
    uint64_t dex_src = 0;
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v4(_occupied_length);

    for (uint32_t i = 0; i < _proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (uint32_t j = 0; j < _proc_dims.x - 1; ++j) {
            tmp_recv = _mm256_load_pd(src + dex_src);
            min_vec4 = _mm256_min_pd(tmp_recv, min_vec4);
            dex_src += 4;
        }
        tmp_recv = _mm256_load_pd(src + dex_src);
        tmp_recv = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(tmp_recv), _shuffle_var));
        min_vec4 = _mm256_min_pd(tmp_recv, min_vec4);
    }

    double res = decx::utils::simd::_mm256d_h_min(min_vec4);
    *res_vec = res;
}




_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec4_fp64_1D(const double* __restrict src, const size_t len, double* res_vec,
    const uint8_t _occupied_length)
{
    __m256d tmp_recv, sum_vec4 = _mm256_load_pd(src);
    const __m256i _shuffle_var = decx::bp::CPUK::extend_shufflevar_v4(_occupied_length);

    for (uint i = 0; i < len - 1; ++i) {
        tmp_recv = _mm256_load_pd(src + ((size_t)i << 2));
        sum_vec4 = _mm256_min_pd(tmp_recv, sum_vec4);
    }
    tmp_recv = _mm256_load_pd(src + ((uint64_t)(len - 1) << 2));
    tmp_recv = _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(tmp_recv), _shuffle_var));
    sum_vec4 = _mm256_min_pd(tmp_recv, sum_vec4);

    double res = decx::utils::simd::_mm256d_h_min(sum_vec4);
    *res_vec = res;
}