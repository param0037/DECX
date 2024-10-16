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


// #include "_mm256_fp32_fp64.h"
#include "typecast_exec_x86_64.h"



_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtps_pd1D(const float* __restrict src, double* __restrict dst, const size_t proc_num)
{
    decx::utils::simd::xmm128_reg recv;
    decx::utils::simd::xmm256_reg store;

    for (int i = 0; i < proc_num; ++i) {
        recv._vf = _mm_load_ps(src + (size_t)i * 4);
        store._vd = _mm256_cvtps_pd(recv._vf);
        _mm256_store_pd(dst + (size_t)i * 4, store._vd);
    }
}




_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtpd_ps1D(const double* __restrict src, float* __restrict dst, const size_t proc_num)
{
    decx::utils::simd::xmm256_reg recv;
    decx::utils::simd::xmm128_reg store;

    for (int i = 0; i < proc_num; ++i) {
        recv._vd = _mm256_load_pd(src + (size_t)i * 4);
        store._vf = _mm256_cvtpd_ps(recv._vd);
        _mm_store_ps(dst + (size_t)i * 4, store._vf);
    }
}




_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtps_pd2D(const float* __restrict     src, 
                                        double* __restrict          dst, 
                                        const uint2                 proc_dims, 
                                        const uint32_t              Wsrc, 
                                        const uint32_t              Wdst)
{
    decx::utils::simd::xmm128_reg recv;
    decx::utils::simd::xmm256_reg store;

    uint32_t dex_src = 0, dex_dst = 0;

    for (int32_t i = 0; i < proc_dims.y; ++i) 
    {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int32_t j = 0; j < proc_dims.x; ++j) {
            recv._vf = _mm_load_ps(src + dex_src);
            store._vd = _mm256_cvtps_pd(recv._vf);

            _mm256_store_pd(dst + dex_dst, store._vd);

            dex_src += 4;
            dex_dst += 4;
        }
    }
}




_THREAD_FUNCTION_ void
decx::type_cast::CPUK::_v256_cvtpd_ps2D(const double* __restrict        src, 
                                        float* __restrict               dst, 
                                        const uint2                     proc_dims, 
                                        const uint32_t                  Wsrc, 
                                        const uint32_t                  Wdst)
{
    decx::utils::simd::xmm256_reg recv;
    decx::utils::simd::xmm128_reg store;

    int32_t dex_src = 0, dex_dst = 0;

    for (int32_t i = 0; i < proc_dims.y; ++i) {
        dex_src = i * Wsrc;
        dex_dst = i * Wdst;
        for (int32_t j = 0; j < proc_dims.x; ++j) {
            recv._vd = _mm256_load_pd(src + dex_src);
            store._vf = _mm256_cvtpd_ps(recv._vd);
            _mm_store_ps(dst + dex_dst, store._vf);

            dex_src += 4;
            dex_dst += 4;
        }
    }
}




// void decx::type_cast::_cvtfp32_fp64_caller1D(const float* src, double* dst, const size_t proc_num)
// {
//     const size_t proc_num_vec4 = proc_num / 4;

//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    
//     const bool _is_MT = (proc_num_vec4 > (1024 * (size_t)t1D.total_thread));

//     if (_is_MT) {
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_num_vec4, t1D.total_thread);

//         const float* loc_src = reinterpret_cast<const float*>(src);
//         double* loc_dst = reinterpret_cast<double*>(dst);

//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtps_pd1D,
//                 loc_src, loc_dst, f_mgr.frag_len);
//             loc_src += ((size_t)f_mgr.frag_len << 2);
//             loc_dst += ((size_t)f_mgr.frag_len << 2);
//         }
//         const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
//             decx::type_cast::CPUK::_v256_cvtps_pd1D, loc_src, loc_dst, _L);

//         t1D.__sync_all_threads();
//     }
//     else {
//         decx::type_cast::CPUK::_v256_cvtps_pd1D(reinterpret_cast<const float*>(src), reinterpret_cast<double*>(dst),
//             proc_num_vec4);
//     }
// }




// void decx::type_cast::_cvtfp64_fp32_caller1D(const double* src, float* dst, const size_t proc_num)
// {
//     const size_t proc_num_vec4 = proc_num / 4;

//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    
//     const bool _is_MT = (proc_num_vec4 > (1024 * (size_t)t1D.total_thread));

//     if (_is_MT) {
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_num_vec4, t1D.total_thread);

//         const double* loc_src = reinterpret_cast<const double*>(src);
//         float* loc_dst = reinterpret_cast<float*>(dst);

//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtpd_ps1D,
//                 loc_src, loc_dst, f_mgr.frag_len);
//             loc_src += ((size_t)f_mgr.frag_len << 2);
//             loc_dst += ((size_t)f_mgr.frag_len << 2);
//         }
//         const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
//             decx::type_cast::CPUK::_v256_cvtpd_ps1D, loc_src, loc_dst, _L);

//         t1D.__sync_all_threads();
//     }
//     else {
//         decx::type_cast::CPUK::_v256_cvtpd_ps1D(reinterpret_cast<const double*>(src), reinterpret_cast<float*>(dst),
//             proc_num_vec4);
//     }
// }





// void decx::type_cast::_cvtfp32_fp64_caller2D(const float* src, double* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst)
// {
//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

//     const bool _is_MT = (proc_dims.y > (1024 * (size_t)t1D.total_thread));

//     if (_is_MT) {
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D.total_thread);

//         const float* loc_src = reinterpret_cast<const float*>(src);
//         double* loc_dst = reinterpret_cast<double*>(dst);

//         const size_t frag_src = Wsrc * f_mgr.frag_len,
//             frag_dst = Wdst * f_mgr.frag_len;

//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtps_pd2D,
//                 loc_src, loc_dst, make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc, Wdst);
//             loc_src += frag_src;
//             loc_dst += frag_dst;
//         }
//         const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
//             decx::type_cast::CPUK::_v256_cvtps_pd2D, loc_src, loc_dst, make_uint2(proc_dims.x, _L), Wsrc, Wdst);

//         t1D.__sync_all_threads();
//     }
//     else {
//         decx::type_cast::CPUK::_v256_cvtps_pd2D(reinterpret_cast<const float*>(src), reinterpret_cast<double*>(dst),
//             make_uint2(proc_dims.x, proc_dims.y), Wsrc, Wdst);
//     }
// }




// void decx::type_cast::_cvtfp64_fp32_caller2D(const double* src, float* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst)
// {
//     decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

//     const bool _is_MT = (proc_dims.y > (1024 * (size_t)t1D.total_thread));

//     if (_is_MT) {
//         decx::utils::frag_manager f_mgr;
//         decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, t1D.total_thread);

//         const double* loc_src = reinterpret_cast<const double*>(src);
//         float* loc_dst = reinterpret_cast<float*>(dst);

//         const size_t frag_src = Wsrc * f_mgr.frag_len,
//             frag_dst = Wdst * f_mgr.frag_len;

//         for (int i = 0; i < t1D.total_thread - 1; ++i) {
//             t1D._async_thread[i] = decx::cpu::register_task_default( decx::type_cast::CPUK::_v256_cvtpd_ps2D,
//                 loc_src, loc_dst, make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc, Wdst);
//             loc_src += frag_src;
//             loc_dst += frag_dst;
//         }
//         const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//         t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(
//             decx::type_cast::CPUK::_v256_cvtpd_ps2D, loc_src, loc_dst, make_uint2(proc_dims.x, _L), Wsrc, Wdst);

//         t1D.__sync_all_threads();
//     }
//     else {
//         decx::type_cast::CPUK::_v256_cvtpd_ps2D(reinterpret_cast<const double*>(src), reinterpret_cast<float*>(dst),
//             make_uint2(proc_dims.x, proc_dims.y), Wsrc, Wdst);
//     }
// }