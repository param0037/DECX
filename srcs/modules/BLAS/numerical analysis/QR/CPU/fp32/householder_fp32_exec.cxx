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

#define _DECX_CPU_PARTS_

#include "../householder_reflector.h"
#include <SIMD/intrinsics_ops.h>
#include <Element_wise/Arithmetics/arithmetic_kernels.h>


namespace decx
{
namespace blas
{
    namespace CPUK{
        _THREAD_FUNCTION_ static void _pow2_sum_v8_fp32(const float* __restrict src, float* __restrict out, const uint32_t proc_len_v8);
    }
}
}


// _THREAD_FUNCTION_ static void
// decx::blas::CPUK::_pow2_sum_v8_fp32(const float* __restrict src, float* __restrict out, const uint32_t proc_len_v8)
// {
//     __m256 res_v8 = _mm256_setzero_ps();

//     for (int32_t i = 0; i < proc_len_v8; ++i){
//         __m256 eles_v8 = _mm256_load_ps(src + (i << 3));
//         res_v8 = _mm256_fmadd_ps(eles_v8, eles_v8, res_v8);
//     }
//     *out = decx::utils::simd::_mm256_h_sum(res_v8);
// }

// void 
// decx::blas::householder_calc_v8_fp32(const float* __restrict p_col, 
//                                      float* __restrict       p_V, 
//                                      const uint32_t          proc_len_v1,
//                                      const uint32_t          local_col_id)
// {
//     const uint32_t proc_len_v8 = decx::blas::CPUK::calc_proc_len_v(local_col_id, 8, proc_len_v1);
//     const float x0 = p_col[0];

//     // Calculate norm2 ^ 2
//     float pow2_sum_post = 0;
//     decx::blas::CPUK::_pow2_sum_v8_fp32(p_col, &pow2_sum_post, proc_len_v8);
//     float x_norm2 = sqrtf(pow2_sum_post);
//     if (x_norm2 < 1e-3){
//         return;
//     }
//     float sign = x0 < 0.f ? -1 : 1;
//     pow2_sum_post -= x0 * x0;
//     float v0 = p_col[local_col_id % 8] - sign * x_norm2;
//     p_col[local_col_id % 8] = v0;
//     pow2_sum_post += v0 * v0;
//     // Normalize and store
//     decx::blas::_subcinv_fp32_exec(p_col, p_V, proc_len_v8, sqrtf(pow2_sum_post));
// }
