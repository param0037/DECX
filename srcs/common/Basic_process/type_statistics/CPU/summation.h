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


#ifndef _CPU_SUMMING_H_
#define _CPU_SUMMING_H_

#include "../../../../modules/core/thread_management/thread_pool.h"
#include "../../../Classes/Vector.h"
#include "../../../Classes/Matrix.h"
#include "../../../Classes/Tensor.h"
#include "../../../FMGR/fragment_arrangment.h"
#include "../../../Classes/classes_util.h"
#include "../../../../modules/core/thread_management/thread_arrange.h"


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
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 8, conc_thr);

    decx::utils::_thread_arrange_1D t1D(conc_thr);
    float* res_arr = new float[conc_thr];
    
    const float* tmp_src = src;
    const uint64_t proc_len = fr_mgr.frag_len * 8;
    for (int i = 0; i < conc_thr - 1; ++i) {
        t1D._async_thread[i] = decx::cpu::register_task_default(
            decx::bp::CPUK::_summing_vec8_fp32, tmp_src, fr_mgr.frag_len, res_arr + i);
        tmp_src += proc_len;
    }
    const uint32_t _L = fr_mgr.is_left ? fr_mgr.frag_left_over : fr_mgr.frag_len;
    t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
        decx::bp::CPUK::_summing_vec8_fp32, tmp_src, _L, res_arr + conc_thr - 1);

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
    const uint conc_thr = decx::cpu::_get_permitted_concurrency();
    decx::utils::frag_manager fr_mgr;
    decx::utils::frag_manager_gen(&fr_mgr, len / 4, conc_thr);

    double* res_arr = new double[conc_thr];
    decx::utils::_thread_arrange_1D t1D(conc_thr);

    const double* tmp_src = src;
    if (fr_mgr.frag_left_over != 0) {
        const size_t proc_len = fr_mgr.frag_len * 4;
        for (int i = 0; i < conc_thr - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(
                decx::bp::CPUK::_summing_vec4_fp64, tmp_src, proc_len / 4, res_arr + i);
            tmp_src += proc_len;
        }
        t1D._async_thread[conc_thr - 1] = decx::cpu::register_task_default(
            decx::bp::CPUK::_summing_vec4_fp64, tmp_src, fr_mgr.frag_left_over, res_arr + conc_thr - 1);
    }
    else {
        const size_t proc_len = fr_mgr.frag_len * 4;
        for (int i = 0; i < conc_thr; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default(
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