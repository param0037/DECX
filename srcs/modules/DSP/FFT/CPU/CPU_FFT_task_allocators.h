/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CPU_FFT_TASK_ALLOCATORS_H_
#define _CPU_FFT_TASK_ALLOCATORS_H_


#include "../../../core/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../classes/classes_util.h"
#include "../fft_utils.h"


namespace decx
{
    namespace signal
    {
        template <class FuncType, uint Radix>
        void FFT_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            FuncType&& f, const float* __restrict src, double* __restrict dst, const size_t signal_length);


        template <class FuncType, uint Radix>
        void FFT_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            FuncType&& f, const double* __restrict src, double* __restrict dst, const size_t signal_length,
            const size_t warp_proc_len);
    }
}


template <class FuncType, uint Radix>
void decx::signal::FFT_assign_task_1D_first(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr, 
    FuncType&& f, const float* __restrict src, double* __restrict dst, const size_t signal_length)
{
    

    const size_t _total_Bop_num = signal_length / Radix;
    if (_total_Bop_num % 4) {
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num, t1D->total_thread);
    }
    else {
        decx::utils::frag_manager_gen(f_mgr, _total_Bop_num / 4, t1D->total_thread);
    }
    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default(f, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(f, src, dst, signal_length, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default(f, src, dst, signal_length, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



template <class FuncType, uint Radix>
void decx::signal::FFT_assign_task_1D(decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
    FuncType&& f, const double* __restrict src, double* __restrict dst, const size_t signal_length,
    const size_t warp_proc_len)
{
    uint2 start_end = make_uint2(0, 0);
    if (f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default(f, src, dst, signal_length, warp_proc_len, start_end);
            start_end.x += f_mgr->frag_len;
        }
        start_end.y += f_mgr->frag_left_over;
        t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default(f, src, dst, signal_length, warp_proc_len, start_end);
    }
    else {
        for (int i = 0; i < t1D->total_thread; ++i) {
            start_end.y += f_mgr->frag_len;
            t1D->_async_thread[i] = decx::cpu::register_task_default(f, src, dst, signal_length, warp_proc_len, start_end);
            start_end.x += f_mgr->frag_len;
        }
    }
    t1D->__sync_all_threads();
}



#endif