/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "module_fp32_exec.h"


_THREAD_FUNCTION_ void
decx::signal::CPUK::_module_fp32_ST(const double* __restrict src, 
                                    float* __restrict dst, 
                                    const size_t _proc_len)       // 4x
{
    __m256 _recv, tmp1;
    __m256d tmp2;

    for (int i = 0; i < _proc_len; ++i) {
        _recv = _mm256_castpd_ps(_mm256_load_pd(src + ((size_t)i << 2)));
        tmp1 = _mm256_mul_ps(_recv, _recv);
        tmp1 = _mm256_hadd_ps(tmp1, tmp1);
        tmp2 = _mm256_permute4x64_pd(_mm256_castps_pd(tmp1), 0b11011000);
        _mm_store_ps(dst + ((size_t)i << 2), _mm256_castps256_ps128(_mm256_castpd_ps(tmp2)));
    }
}



void decx::signal::_module_fp32_caller(const de::CPf* src, float* dst, const size_t _total_len)
{
    const size_t total_len_4x = _total_len / 4;
    const uint conc_thr = (uint)decx::cpI.cpu_concurrency;
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, total_len_4x, conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);

    for (int i = 0; i < conc_thr - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
            decx::signal::CPUK::_module_fp32_ST, (double*)src + ((size_t)f_mgr.frag_len << 2),
            dst + ((size_t)f_mgr.frag_len << 2), f_mgr.frag_len);
    }
    const size_t L_proc = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task(&decx::thread_pool,
        decx::signal::CPUK::_module_fp32_ST, (double*)src + ((size_t)f_mgr.frag_len << 2),
        dst + ((size_t)f_mgr.frag_len << 2), L_proc);

    t1D.__sync_all_threads();
}