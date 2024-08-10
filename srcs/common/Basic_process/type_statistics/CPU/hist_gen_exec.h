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


#ifndef _HIST_GEN_EXEC_H_
#define _HIST_GEN_EXEC_H_


#include "../../../../modules/core/thread_management/thread_arrange.h"
#include "../../../FMGR/fragment_arrangment.h"
#include "../../../../modules/core/thread_management/thread_pool.h"
#include "../../../basic.h"
#include "../../../../modules/core/allocators.h"


namespace decx
{
    namespace bp
    {
        namespace CPUK {
            /**
            * @param proc_dims : ~.x -> width of process area, in vec32
            * ~.y -> height of process area
            * @param Wsrc : Pitch of process matrix, in element (uint8_t)
            * @param _illeagal_space_v32 : The length of idle area in the last vector during the data loading
            */
            _THREAD_FUNCTION_ void 
            _histgen2D_u8_u64(const uint8_t* src, uint64_t* _hist, const uint2 proc_dims, 
                const uint32_t Wsrc, const uint8_t _mask_row_end);
        }

        /**
        * @param proc_dims : ~.x -> width of process area, in element (uint8_t)
        * ~.y -> height of process area
        * @param Wsrc : Pitch of process matrix, in element (uint8_t)
        * @param _illeagal_space_v32 : The length of idle area in the last vector during the data loading
        */
        static void _histgen2D_u8_u64_caller(const uint8_t* src, uint64_t* histogram, const uint2 proc_dims, const uint32_t Wsrc);
    }
}


//
//static void decx::bp::_histgen2D_u8_caller(const uint8_t* src, uint64_t* histogram, const uint2 proc_dims, const uint32_t Wsrc)
//{
//    decx::utils::frag_manager f_mgr;
//    decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, decx::cpu::_get_permitted_concurrency());
//    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
//
//    const uint8_t _leagal_space_v4 = (proc_dims.x % 4) ? (uint8_t)(proc_dims.x - (proc_dims.x / 4) * 4) : 4;
//    
//    __m256i _mask = _mm256_set1_epi64x(1);
//    switch (_leagal_space_v4)
//    {
//    case 1:
//        _mask = _mm256_setr_epi64x(1, 0, 0, 0);
//        break;
//    case 2:
//        _mask = _mm256_setr_epi64x(1, 1, 0, 0);
//        break;
//    case 3:
//        _mask = _mm256_setr_epi64x(1, 1, 1, 0);
//        break;
//    default:
//        break;
//    }
//
//    decx::PtrInfo<uint64_t> _hist_for_threads;
//    if (decx::alloc::_host_virtual_page_malloc(&_hist_for_threads, t1D.total_thread * 256 * 4 * sizeof(uint64_t))) {
//        return;
//    }
//
//    for (int test = 0; test < 10000; ++test) 
//    {
//        const uint8_t* loc_src = src;
//        uint64_t* loc_hist = _hist_for_threads.ptr;
//
//        for (int thread_id = 0; thread_id < t1D.total_thread - 1; ++thread_id) {
//            t1D._async_thread[thread_id] = decx::cpu::register_task_default(decx::bp::CPUK::_histgen2D_u8,
//                loc_src, loc_hist,
//                make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc,
//                _mask);
//
//            loc_src += (Wsrc * f_mgr.frag_len);
//            loc_hist += 256 * 4;
//        }
//        const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
//        t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(decx::bp::CPUK::_histgen2D_u8,
//            loc_src, loc_hist,
//            make_uint2(proc_dims.x, _L), Wsrc,
//            _mask);
//
//        t1D.__sync_all_threads();
//
//        /*__m256i accu, recv;
//        if (t1D.total_thread > 1) {
//            for (uint32_t i = 0; i < 256; ++i) {
//                accu = _mm256_load_si256((__m256i*)_hist_for_threads.ptr + i);
//                for (int threadId = 1; threadId < t1D.total_thread; ++threadId) {
//                    recv = _mm256_load_si256((__m256i*)_hist_for_threads.ptr + i + threadId * 256);
//                    accu = _mm256_add_epi64(accu, recv);
//                }
//                _mm256_store_si256((__m256i*)_hist_for_threads.ptr + i, accu);
//            }
//        }
//        for (int i = 0; i < 256; ++i) {
//            recv = _mm256_load_si256((__m256i*)_hist_for_threads.ptr + i);
//            histogram[i] =  decx::utils::simd::_mm256i_h_sum_epi64(recv);
//        }*/
//    }
//    decx::alloc::_host_virtual_page_dealloc(&_hist_for_threads);
//}
//



static void decx::bp::_histgen2D_u8_u64_caller(const uint8_t* src, uint64_t* histogram, const uint2 proc_dims, const uint32_t Wsrc)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, decx::cpu::_get_permitted_concurrency());
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    const uint8_t _leagal_space_v4 = (proc_dims.x % 4) ? (uint8_t)(proc_dims.x % 4) : 4;
    
    decx::PtrInfo<uint64_t> _hist_for_threads;
    if (decx::alloc::_host_virtual_page_malloc(&_hist_for_threads, t1D.total_thread * 256 * sizeof(uint64_t))) {
        return;
    }

    const uint8_t* loc_src = src;
    uint64_t* loc_hist = _hist_for_threads.ptr;

    for (int thread_id = 0; thread_id < t1D.total_thread - 1; ++thread_id) {
        t1D._async_thread[thread_id] = decx::cpu::register_task_default(decx::bp::CPUK::_histgen2D_u8_u64,
            loc_src, loc_hist,
            make_uint2(proc_dims.x, f_mgr.frag_len), Wsrc,
            _leagal_space_v4);

        loc_src += (Wsrc * f_mgr.frag_len);
        loc_hist += 256;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default(decx::bp::CPUK::_histgen2D_u8_u64,
        loc_src, loc_hist,
        make_uint2(proc_dims.x, _L), Wsrc,
        _leagal_space_v4);

    t1D.__sync_all_threads();

    __m256i accu, recv;
    if (t1D.total_thread > 1) {
        for (uint32_t i = 0; i < 256 / 4; ++i) {
            accu = _mm256_load_si256((__m256i*)_hist_for_threads.ptr + i);
            for (int threadId = 1; threadId < t1D.total_thread; ++threadId) {
                recv = _mm256_load_si256((__m256i*)_hist_for_threads.ptr + i + threadId * 256 / 4);
                accu = _mm256_add_epi64(accu, recv);
            }
            _mm256_store_si256((__m256i*)histogram + i, accu);
        }
    }
    decx::alloc::_host_virtual_page_dealloc(&_hist_for_threads);
}




#endif