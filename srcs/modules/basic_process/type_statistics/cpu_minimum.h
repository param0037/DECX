/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CPU_MIMIMUM_H_
#define _CPU_MIMIMUM_H_

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
            _THREAD_FUNCTION_ void
                _minimum_vec8_fp32(const float* src, const size_t len, float* res_vec);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            _THREAD_FUNCTION_ void
                _minimum_vec4_fp64(const double* src, const size_t len, double* res_vec);
        }

        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        static void _minimum_fp32_1D_caller(const float* src, const size_t len, float* res_vec);


        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        static void _minimum_fp64_1D_caller(const double* src, const size_t len, double* res_vec);
    }
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec8_fp32(const float* src, const size_t len, float* res_vec)
{
    __m256 tmp_recv, sum_vec8 = _mm256_set1_ps(0);
    float res_vec8[8], res;

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_ps(src + ((size_t)i << 3));
        sum_vec8 = _mm256_min_ps(tmp_recv, sum_vec8);
    }

    _mm256_store_ps(res_vec8, sum_vec8);
    res = res_vec8[0];
#pragma unroll 7
    for (int j = 1; j < 7; ++j) {
        res = res > res_vec8[j] ? res_vec8[j] : res;
    }
    *res_vec = res;
}



_THREAD_FUNCTION_ void
decx::bp::CPUK::_minimum_vec4_fp64(const double* src, const size_t len, double* res_vec)
{
    __m256d tmp_recv, sum_vec4 = _mm256_set1_pd(0);
    double res_vec4[4], res = 0;;

    for (uint i = 0; i < len; ++i) {
        tmp_recv = _mm256_load_pd(src + ((size_t)i << 3));
        sum_vec4 = _mm256_min_pd(tmp_recv, sum_vec4);
    }

    _mm256_store_pd(res_vec4, sum_vec4);
    res = res_vec4[0];
#pragma unroll 3
    for (int j = 1; j < 4; ++j) {
        res = res > res_vec4[j] ? res_vec4[j] : res;
    }
    *res_vec = res;
}



static void decx::bp::_minimum_fp32_1D_caller(const float* src, const size_t len, float* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpI.cpu_concurrency;
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 8, conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    float* res_arr = new float[conc_thr];

    const float* tmp_src = src;
    if (fr_mgr.frag_left_over != 0) {
        const size_t proc_len = fr_mgr.frag_len * 8;
        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_minimum_vec8_fp32, tmp_src, proc_len / 8, res_arr + i);
            tmp_src += proc_len;
        }
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::bp::CPUK::_minimum_vec8_fp32, tmp_src, fr_mgr.frag_left_over, res_arr + conc_thr - 1);
    }
    else {
        const size_t proc_len = fr_mgr.frag_len * 8;
        for (int i = 0; i < conc_thr; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_minimum_vec8_fp32, tmp_src, proc_len / 8, res_arr + i);
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



static void decx::bp::_minimum_fp64_1D_caller(const double* src, const size_t len, double* res_vec)
{
    // the number of available concurrent threads
    const uint conc_thr = decx::cpI.cpu_concurrency;
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 4, conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    double* res_arr = new double[conc_thr];

    const double* tmp_src = src;
    if (fr_mgr.frag_left_over != 0) {
        const size_t proc_len = fr_mgr.frag_len * 4;
        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_minimum_vec4_fp64, tmp_src, proc_len / 4, res_arr + i);
            tmp_src += proc_len;
        }
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task(&decx::thread_pool,
            decx::bp::CPUK::_minimum_vec4_fp64, tmp_src, fr_mgr.frag_left_over, res_arr + conc_thr - 1);
    }
    else {
        const size_t proc_len = fr_mgr.frag_len * 4;
        for (int i = 0; i < conc_thr; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task(&decx::thread_pool,
                decx::bp::CPUK::_minimum_vec4_fp64, tmp_src, proc_len / 4, res_arr + i);
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
        _DECX_API_ de::DH Min_fp32(de::Vector& src, float* res);


        _DECX_API_ de::DH Min_fp32(de::Matrix& src, float* res);


        _DECX_API_ de::DH Min_fp32(de::Tensor& src, float* res);


        _DECX_API_ de::DH Min_fp64(de::Vector& src, double* res);


        _DECX_API_ de::DH Min_fp64(de::Matrix& src, double* res);


        _DECX_API_ de::DH Min_fp64(de::Tensor& src, double* res);
    }
}



#endif