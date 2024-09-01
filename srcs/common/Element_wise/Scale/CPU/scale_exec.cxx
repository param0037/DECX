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


#include "../normalization.h"


_THREAD_FUNCTION_ void decx::CPUK::
normalize_scale_v8_fp32(const float* __restrict src, 
                        float* __restrict       dst, 
                        const uint64_t          proc_len_v, 
                        const double2           min_max, 
                        const double2           range)
{
    const __m256 min_max_dst_rct = _mm256_set1_ps(1.f / (min_max.y - min_max.x));
    const __m256 range_dst = _mm256_set1_ps(range.y - range.x);
    __m256 recv, store;

    for (uint32_t i = 0; i < proc_len_v; ++i) {
        recv = _mm256_load_ps(src + (i << 3));

        recv = _mm256_sub_ps(recv, _mm256_set1_ps(min_max.x));
        recv = _mm256_div_ps(recv, min_max_dst_rct);
        store = _mm256_fmadd_ps(recv, range_dst, _mm256_set1_ps(range.x));

        _mm256_store_ps(dst + (i << 3), store);
    }
}


_THREAD_FUNCTION_ void decx::CPUK::
normalize_scale_v4_fp64(const double* __restrict    src, 
                        double* __restrict          dst, 
                        const uint64_t              proc_len_v, 
                        const double2               min_max, 
                        const double2               range)
{
    const __m256d min_max_dst_rct = _mm256_set1_pd(1.0 / (min_max.y - min_max.x));
    const __m256d range_dst = _mm256_set1_pd(range.y - range.x);
    __m256d recv, store;

    for (uint32_t i = 0; i < proc_len_v; ++i) {
        recv = _mm256_load_pd(src + (i << 2));

        recv = _mm256_sub_pd(recv, _mm256_set1_pd(min_max.x));
        recv = _mm256_mul_pd(recv, min_max_dst_rct);
        store = _mm256_fmadd_pd(recv, range_dst, _mm256_set1_pd(range.x));

        _mm256_store_pd(dst + (i << 2), store);
    }
}
