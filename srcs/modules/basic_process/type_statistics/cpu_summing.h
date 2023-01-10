/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CPU_SUMMING_H_
#define _CPU_SUMMING_H_

#include "../../core/thread_management/thread_pool.h"
#include "../../classes/Vector.h"
#include "../../classes/Matrix.h"
#include "../../classes/Tensor.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../classes/classes_util.h"


namespace decx
{
    namespace bp {
        namespace CPUK {
            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            _THREAD_FUNCTION_ static void
                _summing_vec8_fp32(const float* src, const size_t len, float* res_vec);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256d
            * @param res_vec : the result vector in __m256d
            */
            _THREAD_FUNCTION_ static void
                _summing_vec4_fp64(const double* src, const size_t len, double* res_vec);
        }

        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        static void _summing_fp32_1D_caller(const float* src, const size_t len, float* res_vec);


        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        static void _summing_fp64_1D_caller(const double* src, const size_t len, double* res_vec);
    }
}



_THREAD_FUNCTION_ static void
decx::bp::CPUK::_summing_vec8_fp32(const float* src, const size_t len, float* res_vec)
{
    __m256 tmp_recv, sum_vec8 = _mm256_set1_ps(0);

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_ps(src + ((size_t)i << 3));
        sum_vec8 = _mm256_add_ps(tmp_recv, sum_vec8);
    }
    
    *res_vec = decx::utils::simd::_mm256_h_sum(sum_vec8);
}


_THREAD_FUNCTION_ static void
decx::bp::CPUK::_summing_vec4_fp64(const double* src, const size_t len, double* res_vec)
{
    __m256d tmp_recv, sum_vec8 = _mm256_set1_pd(0);

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_pd(src + ((size_t)i << 2));
        sum_vec8 = _mm256_add_pd(tmp_recv, sum_vec8);
    }

    *res_vec = decx::utils::simd::_mm256d_h_sum(sum_vec8);
}



static void decx::bp::_summing_fp32_1D_caller(const float* src, const size_t len, float* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpI.cpu_concurrency;
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 8, conc_thr);

    //std::future<void>* fut = new std::future<void>[conc_thr];
    decx::utils::_thread_arrange_1D t1D(conc_thr);
    float* res_arr = new float[conc_thr];
    
    const float* tmp_src = src;
    if (fr_mgr.frag_left_over != 0) {
        const size_t proc_len = fr_mgr.frag_len * 8;
        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_summing_vec8_fp32, tmp_src, proc_len / 8, res_arr + i);
            tmp_src += proc_len;
        }
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::bp::CPUK::_summing_vec8_fp32, tmp_src, fr_mgr.frag_left_over, res_arr + conc_thr - 1);
    }
    else {
        const size_t proc_len = fr_mgr.frag_len * 8;
        for (int i = 0; i < conc_thr; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_summing_vec8_fp32, tmp_src, proc_len / 8, res_arr + i);
            tmp_src += proc_len;
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



static void decx::bp::_summing_fp64_1D_caller(const double* src, const size_t len, double* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpI.cpu_concurrency;
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 4, conc_thr);

    //std::future<void>* fut = new std::future<void>[conc_thr];
    double* res_arr = new double[conc_thr];
    decx::utils::_thread_arrange_1D t1D(conc_thr);

    const double* tmp_src = src;
    if (fr_mgr.frag_left_over != 0) {
        const size_t proc_len = fr_mgr.frag_len * 4;
        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_summing_vec4_fp64, tmp_src, proc_len / 4, res_arr + i);
            tmp_src += proc_len;
        }
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::bp::CPUK::_summing_vec4_fp64, tmp_src, fr_mgr.frag_left_over, res_arr + conc_thr - 1);
    }
    else {
        const size_t proc_len = fr_mgr.frag_len * 4;
        for (int i = 0; i < conc_thr; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_summing_vec4_fp64, tmp_src, proc_len / 4, res_arr + i);
            tmp_src += proc_len;
        }
    }

    t1D.__sync_all_threads();

    double res = 0;
    for (int i = 0; i < conc_thr; ++i) {
        res += res_arr[i];
    }

    *res_vec = res;

    delete[] res_arr;
}




namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Sum_fp32(de::Vector& src, float* res);


        _DECX_API_ de::DH Sum_fp32(de::Matrix& src, float* res);


        _DECX_API_ de::DH Sum_fp32(de::Tensor& src, float* res);



        _DECX_API_ de::DH Sum_fp64(de::Vector& src, double* res);


        _DECX_API_ de::DH Sum_fp64(de::Matrix& src, double* res);


        _DECX_API_ de::DH Sum_fp64(de::Tensor& src, double* res);
    }
}


#endif