/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _OPERATORS_FRAME_EXEC_H_
#define _OPERATORS_FRAME_EXEC_H_

#include "../../core/thread_management/thread_pool.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../core/utils/fragment_arrangment.h"


namespace decx
{
    namespace calc {
        typedef void _fp32_binary_ops_m(const float*, float*, float*, size_t);
        typedef void _fp32_binary_ops_c(const float*, const float, float*, size_t);
        typedef void _int32_binary_ops_m(const __m256i*, __m256i*, __m256i*, size_t);
        typedef void _int32_binary_ops_c(const __m256i*, const int, __m256i*, size_t);
        typedef void _fp64_binary_ops_m(const double*, double*, double*, size_t);
        typedef void _fp64_binary_ops_c(const double*, const double, double*, size_t);


        typedef void _fp32_ternary_ops_m(const float*, float*, float*, float*, size_t);
        typedef void _fp32_ternary_ops_c(const float*, const float, float*, float*, size_t);
        typedef void _int32_ternary_ops_m(const __m256i*, __m256i*, __m256i*, __m256i*, size_t);
        typedef void _int32_ternary_ops_c(const __m256i*, const int, __m256i*, __m256i*, size_t);
        typedef void _fp64_ternary_ops_m(const double*, double*, double*, double*, size_t);
        typedef void _fp64_ternary_ops_c(const double*, const double, double*, double*, size_t);


        /**
        * @param A : pointer of sub-matrix A
        * @param B : pointer of sub-matrix B
        * @param f_mgr : of which length is in __m256 (float x8)
        */
        static void operators_caller_m_fp32(decx::calc::_fp32_binary_ops_m _op, float* A, float* B, float* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_m_int32(decx::calc::_int32_binary_ops_m _op, int32_t* A, int32_t* B, int32_t* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_m_fp64(decx::calc::_fp64_binary_ops_m _op, double* A, double* B, double* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_m_fp32(decx::calc::_fp32_ternary_ops_m _op, float* A, float* B, float* C, float* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_m_int32(decx::calc::_int32_ternary_ops_m _op, int32_t* A, int32_t* B, int32_t* C, int32_t* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_m_fp64(decx::calc::_fp64_ternary_ops_m _op, double* A, double* B, double* C, double* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        // ---------------------------------------------------- c -------------------------------------------------------


        static void operators_caller_c_fp32(decx::calc::_fp32_binary_ops_c _op, float* src, const float B, float* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_c_int32(decx::calc::_int32_binary_ops_c _op, int32_t* src, const int32_t B, int32_t* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_c_fp64(decx::calc::_fp64_binary_ops_c _op, double* src, const double B, double* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_c_fp32(decx::calc::_fp32_ternary_ops_c _op, float* A, const float __x, float* B, float* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_c_int32(decx::calc::_int32_ternary_ops_c _op, int32_t* A, const int32_t __x, int32_t* B, int32_t* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);


        static void operators_caller_c_fp64(decx::calc::_fp64_ternary_ops_c _op, double* A, const double __x, double* B, double* dst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}



void decx::calc::operators_caller_m_fp32(decx::calc::_fp32_binary_ops_m _op, float* A, float* B, float* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 3) * f_mgr->frag_len,
                    B + (i << 3) * f_mgr->frag_len,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 3) * f_mgr->frag_len,
                    B + (i << 3) * f_mgr->frag_len,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                A + (f_mgr->frag_len << 3),
                B + (f_mgr->frag_len << 3),
                dst + (f_mgr->frag_len << 3),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}



void decx::calc::operators_caller_m_int32(decx::calc::_int32_binary_ops_m _op, int32_t* A, int32_t* B, int32_t* dst, 
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op, 
                    (__m256i*)A + i * f_mgr->frag_len,
                    (__m256i*)B + i * f_mgr->frag_len,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op, 
                    (__m256i*)A + i * f_mgr->frag_len,
                    (__m256i*)B + i * f_mgr->frag_len,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op, 
                (__m256i*)A + f_mgr->frag_len,
                (__m256i*)B + f_mgr->frag_len,
                (__m256i*)dst + f_mgr->frag_len,
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}



void decx::calc::operators_caller_m_fp64(decx::calc::_fp64_binary_ops_m _op, double* A, double* B, double* dst, 
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 2) * f_mgr->frag_len,
                    B + (i << 2) * f_mgr->frag_len,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 2) * f_mgr->frag_len,
                    B + (i << 2) * f_mgr->frag_len,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                A + (f_mgr->frag_len << 2),
                B + (f_mgr->frag_len << 2),
                dst + (f_mgr->frag_len << 2),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}



void decx::calc::operators_caller_m_fp32(decx::calc::_fp32_ternary_ops_m _op, float* A, float* B, float* C, float* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 3) * f_mgr->frag_len,
                    B + (i << 3) * f_mgr->frag_len,
                    C + (i << 3) * f_mgr->frag_len,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 3) * f_mgr->frag_len,
                    B + (i << 3) * f_mgr->frag_len,
                    C + (i << 3) * f_mgr->frag_len,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                A + (f_mgr->frag_len << 3),
                B + (f_mgr->frag_len << 3),
                C + (f_mgr->frag_len << 3),
                dst + (f_mgr->frag_len << 3),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}



void decx::calc::operators_caller_m_int32(decx::calc::_int32_ternary_ops_m _op, int32_t* A, int32_t* B, int32_t* C, int32_t* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    (__m256i*)A + i * f_mgr->frag_len,
                    (__m256i*)B + i * f_mgr->frag_len,
                    (__m256i*)C + i * f_mgr->frag_len,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    (__m256i*)A + i * f_mgr->frag_len,
                    (__m256i*)B + i * f_mgr->frag_len,
                    (__m256i*)C + i * f_mgr->frag_len,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                (__m256i*)A + f_mgr->frag_len,
                (__m256i*)B + f_mgr->frag_len,
                (__m256i*)C + f_mgr->frag_len,
                (__m256i*)dst + f_mgr->frag_len,
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}



void decx::calc::operators_caller_m_fp64(decx::calc::_fp64_ternary_ops_m _op, double* A, double* B, double* C, double* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 2) * f_mgr->frag_len,
                    B + (i << 2) * f_mgr->frag_len,
                    C + (i << 2) * f_mgr->frag_len,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 2) * f_mgr->frag_len,
                    B + (i << 2) * f_mgr->frag_len,
                    C + (i << 2) * f_mgr->frag_len,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                A + (f_mgr->frag_len << 2),
                B + (f_mgr->frag_len << 2),
                C + (f_mgr->frag_len << 2),
                dst + (f_mgr->frag_len << 2),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}


// ----------------------------------------------------- c ------------------------------------------------------------


void decx::calc::operators_caller_c_fp32(decx::calc::_fp32_binary_ops_c _op, float* src, const float __x, float* dst, 
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    src + (i << 3) * f_mgr->frag_len,
                    __x,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    src + (i << 3) * f_mgr->frag_len,
                    __x,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                src + (f_mgr->frag_len << 3),
                __x,
                dst + (f_mgr->frag_len << 3),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}




void decx::calc::operators_caller_c_int32(decx::calc::_int32_binary_ops_c _op, int* src, const int __x, int* dst, 
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    (__m256i*)src + i * f_mgr->frag_len,
                    __x,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    (__m256i*)src + i * f_mgr->frag_len,
                    __x,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                (__m256i*)src + f_mgr->frag_len,
                __x,
                (__m256i*)dst + f_mgr->frag_len,
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}




void decx::calc::operators_caller_c_fp64(decx::calc::_fp64_binary_ops_c _op, double* src, const double __x, double* dst, 
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    src + (i << 2) * f_mgr->frag_len,
                    __x,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    src + (i << 2) * f_mgr->frag_len,
                    __x,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                src + (f_mgr->frag_len << 2),
                __x,
                dst + (f_mgr->frag_len << 2),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}





void decx::calc::operators_caller_c_fp32(decx::calc::_fp32_ternary_ops_c _op, float* A, const float __x, float* B, float* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 3) * f_mgr->frag_len,
                    __x,
                    B + (i << 3) * f_mgr->frag_len,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 3) * f_mgr->frag_len,
                    __x,
                    B + (i << 3) * f_mgr->frag_len,
                    dst + (i << 3) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                A + (f_mgr->frag_len << 3),
                __x,
                B + (f_mgr->frag_len << 3),
                dst + (f_mgr->frag_len << 3),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}




void decx::calc::operators_caller_c_int32(decx::calc::_int32_ternary_ops_c _op, int32_t* A, const int32_t __x, int32_t* B, int32_t* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    (__m256i*)A + i * f_mgr->frag_len,
                    __x,
                    (__m256i*)B + i * f_mgr->frag_len,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    (__m256i*)A + i * f_mgr->frag_len,
                    __x,
                    (__m256i*)B + i * f_mgr->frag_len,
                    (__m256i*)dst + i * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                (__m256i*)A + f_mgr->frag_len,
                __x,
                (__m256i*)B + f_mgr->frag_len,
                (__m256i*)dst + f_mgr->frag_len,
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}




void decx::calc::operators_caller_c_fp64(decx::calc::_fp64_ternary_ops_c _op, double* A, const double __x, double* B, double* dst,
    decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr)
{
    if (!f_mgr->is_left) {
        for (int i = 0; i < t1D->total_thread; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 2) * f_mgr->frag_len,
                    __x,
                    B + (i << 2) * f_mgr->frag_len,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
    }
    else {
        for (int i = 0; i < t1D->total_thread - 1; ++i) {
            t1D->_async_thread[i] =
                decx::cpu::register_task(&decx::thread_pool, _op,
                    A + (i << 2) * f_mgr->frag_len,
                    __x,
                    B + (i << 2) * f_mgr->frag_len,
                    dst + (i << 2) * f_mgr->frag_len,
                    f_mgr->frag_len);
        }
        t1D->_async_thread[t1D->total_thread - 1] =
            decx::cpu::register_task(&decx::thread_pool, _op,
                A + (f_mgr->frag_len << 2),
                __x,
                B + (f_mgr->frag_len << 2),
                dst + (f_mgr->frag_len << 2),
                f_mgr->frag_left_over);
    }

    t1D->__sync_all_threads();
}




#endif