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

#include "cpu_dot_fp32.h"
#include "../../../../core/thread_management/thread_arrange.h"


_THREAD_FUNCTION_ void
decx::dot::CPUK::_dot_vec8_fp32(const float* A, const float* B, const size_t len, float* res_vec)
{
    __m256 tmp_recv1, tmp_recv2, sum_vec8 = _mm256_set1_ps(0);

    for (uint i = 0; i < len; ++i) {
        tmp_recv1 = _mm256_load_ps(A + ((size_t)i << 3));
        tmp_recv2 = _mm256_load_ps(B + ((size_t)i << 3));
        sum_vec8 = _mm256_fmadd_ps(tmp_recv1, tmp_recv2, sum_vec8);
    }

    *res_vec = decx::utils::simd::_mm256_h_sum(sum_vec8);
}


void decx::dot::_dot_fp32_1D_caller(const float* A, const float* B, const size_t len, float* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 8, conc_thr);
    decx::utils::_thread_arrange_1D t1D(conc_thr);

    float* res_arr = new float[conc_thr];

    const float* tmp_A_ptr = A, * tmp_B_ptr = B;
    if (fr_mgr.frag_left_over != 0) {
        const size_t proc_len = fr_mgr.frag_len * 8;
        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(
                decx::dot::CPUK::_dot_vec8_fp32, tmp_A_ptr, tmp_B_ptr, proc_len / 8, res_arr + i);
            tmp_A_ptr += proc_len;
            tmp_B_ptr += proc_len;
        }
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
            decx::dot::CPUK::_dot_vec8_fp32, tmp_A_ptr, tmp_B_ptr, fr_mgr.frag_left_over, res_arr + conc_thr - 1);
    }
    else {
        const size_t proc_len = fr_mgr.frag_len * 8;
        for (int i = 0; i < conc_thr; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(
                decx::dot::CPUK::_dot_vec8_fp32, tmp_A_ptr, tmp_B_ptr, proc_len / 8, res_arr + i);
            tmp_A_ptr += proc_len;
            tmp_B_ptr += proc_len;
        }
    }

    t1D.__sync_all_threads();

    float res = 0;
    for (int i = 0; i < conc_thr; ++i) {
        res += res_arr[i];
    }

    *res_vec = res;

    delete[] res_arr;
}
