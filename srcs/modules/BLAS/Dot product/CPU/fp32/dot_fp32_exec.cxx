/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
