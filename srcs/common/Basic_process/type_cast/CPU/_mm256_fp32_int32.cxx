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

#include "_mm256_fp32_int32.h"


_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtps_i32(const float* __restrict src, float* __restrict dst, const size_t proc_num)
{
    decx::utils::simd::xmm256_reg recv, store;

    for (int i = 0; i < proc_num; ++i) {
        recv._vf = _mm256_load_ps(src + (size_t)i * 8);
        store._vi = _mm256_cvtps_epi32(recv._vf);
        _mm256_store_ps(dst + (size_t)i * 8, store._vf);
    }
}




_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvti32_ps(const float* __restrict src, float* __restrict dst, const size_t proc_num)
{
    decx::utils::simd::xmm256_reg recv, store;

    for (int i = 0; i < proc_num; ++i) {
        recv._vf = _mm256_load_ps(src + (size_t)i * 8);
        store._vf = _mm256_cvtepi32_ps(recv._vi);
        _mm256_store_ps(dst + (size_t)i * 8, store._vf);
    }
}





void decx::type_cast::_cvtfp32_i32_caller(const float* src, float* dst, const size_t proc_num)
{
    const size_t proc_num_vec4 = proc_num / 8;

    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    
    const bool _is_MT = (proc_num_vec4 > (1024 * (size_t)t1D.total_thread));

    if (_is_MT) {
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, proc_num_vec4, t1D.total_thread);

        const float* loc_src = reinterpret_cast<const float*>(src);
        float* loc_dst = reinterpret_cast<float*>(dst);

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtps_i32,
                loc_src, loc_dst, f_mgr.frag_len);
            loc_src += ((size_t)f_mgr.frag_len << 3);
            loc_dst += ((size_t)f_mgr.frag_len << 3);
        }
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
            decx::type_cast::CPUK::_v256_cvtps_i32, loc_src, loc_dst, f_mgr.frag_len);

        t1D.__sync_all_threads();
    }
    else {
        decx::type_cast::CPUK::_v256_cvtps_i32(reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst),
            proc_num_vec4);
    }
}




void decx::type_cast::_cvti32_fp32_caller(const float* src, float* dst, const size_t proc_num)
{
    const size_t proc_num_vec4 = proc_num / 8;

    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    
    const bool _is_MT = (proc_num_vec4 > (1024 * (size_t)t1D.total_thread));

    if (_is_MT) {
        decx::utils::frag_manager f_mgr;
        decx::utils::frag_manager_gen(&f_mgr, proc_num_vec4, t1D.total_thread);

        const float* loc_src = reinterpret_cast<const float*>(src);
        float* loc_dst = reinterpret_cast<float*>(dst);

        for (int i = 0; i < t1D.total_thread - 1; ++i) {
            t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvti32_ps,
                loc_src, loc_dst, f_mgr.frag_len);
            loc_src += ((size_t)f_mgr.frag_len << 3);
            loc_dst += ((size_t)f_mgr.frag_len << 3);
        }
        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
            decx::type_cast::CPUK::_v256_cvti32_ps, loc_src, loc_dst, f_mgr.frag_len);

        t1D.__sync_all_threads();
    }
    else {
        decx::type_cast::CPUK::_v256_cvti32_ps(reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst),
            proc_num_vec4);
    }
}