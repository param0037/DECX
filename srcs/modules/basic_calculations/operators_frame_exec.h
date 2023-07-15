/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _OPERATORS_FRAME_EXEC_H_
#define _OPERATORS_FRAME_EXEC_H_

#include "../core/thread_management/thread_pool.h"
#include "../core/thread_management/thread_arrange.h"
#include "../core/utils/fragment_arrangment.h"


namespace decx
{
    namespace calc 
    {
        typedef void _fp32_single_ops       (const float*, float*, uint64_t);
        typedef void _fp32_clip_ops         (const float*, const float2, float*, uint64_t);
        typedef void _fp64_clip_ops         (const double*, const double2, double*, uint64_t);
        typedef void _fp64_single_ops       (const double*, double*, uint64_t);

        typedef void _fp32_binary_ops_m     (const float*, const float*, float*, uint64_t);
        typedef void _fp32_binary_ops_c     (const float*, const float, float*, uint64_t);
        typedef void _int32_binary_ops_m    (const int*, const int*, int*, uint64_t);
        typedef void _int32_binary_ops_c    (const int*, const int, int*, uint64_t);
        typedef void _fp64_binary_ops_m     (const double*, const double*, double*, uint64_t);
        typedef void _fp64_binary_ops_c     (const double*, const double, double*, uint64_t);

        typedef void _fp32_ternary_ops_m    (const float*, const float*, const float*, float*, uint64_t);
        typedef void _fp32_ternary_ops_c    (const float*, const float, const float*, float*, uint64_t);
        typedef void _int32_ternary_ops_m   (const int*, const int*, const int*, int*, uint64_t);
        typedef void _int32_ternary_ops_c   (const int*, const int, const int*, int*, uint64_t);
        typedef void _fp64_ternary_ops_m    (const double*, const double*, const double*, double*, uint64_t);
        typedef void _fp64_ternary_ops_c    (const double*, const double, const double*, double*, uint64_t);


        template <typename T_op, typename T_data, uint8_t _align>
        static void operators_caller(T_op _op, T_data* src, T_data* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);

        template <typename T_op, typename T_data, uint8_t _align>
        static void operators_caller_m(T_op _op, T_data* A, T_data* B, T_data* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        template <typename T_op, typename T_data, uint8_t _align>
        static void operators_caller_m3(T_op _op, T_data* A, T_data* B, T_data* C, T_data* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        template <typename T_op, typename T_data, uint8_t _align, typename T_C = T_data>
        static void operators_caller_c(T_op _op, T_data* src, const T_C __x, T_data* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        template <typename T_op, typename T_data, uint8_t _align>
        static void operators_caller_c3(T_op _op, T_data* src, const T_data __x, T_data* C, T_data* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}


template <typename T_op, typename T_data, uint8_t _align>
static void decx::calc::operators_caller(T_op _op,
                                         T_data* src, T_data* dst,
                                         decx::utils::_thr_1D* t1D,
                                         decx::utils::frag_manager* f_mgr)
{
    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( _op,
            src + (i * _align * f_mgr->frag_len),
            dst + (i * _align * f_mgr->frag_len), f_mgr->frag_len);
    }
    const uint64_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( _op,
        src + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
        dst + (f_mgr->frag_len * _align * (t1D->total_thread - 1)), _L);

    t1D->__sync_all_threads();
}


template <typename T_op, typename T_data, uint8_t _align>
static void decx::calc::operators_caller_m3(T_op _op, 
                                           T_data* A, T_data* B, T_data* C, T_data* dst, 
                                           decx::utils::_thr_1D* t1D, 
                                           decx::utils::frag_manager* f_mgr)
{
    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( _op,
            A + (i * _align * f_mgr->frag_len),
            B + (i * _align * f_mgr->frag_len),
            C + (i * _align * f_mgr->frag_len),
            dst + (i * _align * f_mgr->frag_len), f_mgr->frag_len);
    }
    const uint64_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( _op,
        A + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
        B + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
        C + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
        dst + (f_mgr->frag_len * _align * (t1D->total_thread - 1)), _L);

    t1D->__sync_all_threads();
}



template <typename T_op, typename T_data, uint8_t _align>
static void decx::calc::operators_caller_m(T_op _op,
    T_data* A, T_data* B, T_data* dst,
    decx::utils::_thr_1D* t1D,
    decx::utils::frag_manager* f_mgr)
{
    for (int i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default( _op,
            A + (i * _align * f_mgr->frag_len),
            B + (i * _align * f_mgr->frag_len),
            dst + (i * _align * f_mgr->frag_len), f_mgr->frag_len);
    }
    const uint64_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] = decx::cpu::register_task_default( _op,
        A + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
        B + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
        dst + (f_mgr->frag_len * _align * (t1D->total_thread - 1)), _L);

    t1D->__sync_all_threads();
}



template <typename T_op, typename T_data, uint8_t _align, typename T_C>
static void decx::calc::operators_caller_c(T_op _op, T_data* src, const T_C __x, T_data* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    for (uint32_t i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] =
            decx::cpu::register_task_default( _op,
                src + (i * _align * f_mgr->frag_len), __x, 
                dst + (i * _align * f_mgr->frag_len), f_mgr->frag_len);
    }
    const uint64_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] =
        decx::cpu::register_task_default( _op,
            src + ((t1D->total_thread - 1) * f_mgr->frag_len * _align), __x,
            dst + ((t1D->total_thread - 1) * f_mgr->frag_len * _align), _L);

    t1D->__sync_all_threads();
}




template <typename T_op, typename T_data, uint8_t _align>
static void decx::calc::operators_caller_c3(T_op _op, T_data* A, const T_data __x, T_data* B, T_data* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    for (uint32_t i = 0; i < t1D->total_thread - 1; ++i) {
        t1D->_async_thread[i] =
            decx::cpu::register_task_default( _op,
                A + (i * _align) * f_mgr->frag_len, __x, 
                B + (i * _align) * f_mgr->frag_len,
                dst + (i * _align) * f_mgr->frag_len, f_mgr->frag_len);
    }
    const uint64_t _L = f_mgr->is_left ? f_mgr->frag_left_over : f_mgr->frag_len;
    t1D->_async_thread[t1D->total_thread - 1] =
        decx::cpu::register_task_default( _op,
            A + (f_mgr->frag_len * _align * (t1D->total_thread - 1)), __x,
            B + (f_mgr->frag_len * _align * (t1D->total_thread - 1)),
            dst + (f_mgr->frag_len * _align * (t1D->total_thread - 1)), _L);

    t1D->__sync_all_threads();
}




#endif