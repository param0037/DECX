/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _HIST_GEN_EXEC_H_
#define _HIST_GEN_EXEC_H_


#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/allocators.h"


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
            _histgen2D_u8(const uint8_t* src, uint64_t* _hist, const uint2 proc_dims, 
                const uint32_t Wsrc, const uint8_t _illeagal_space_v32);
        }

        /**
        * @param proc_dims : ~.x -> width of process area, in element (uint8_t)
        * ~.y -> height of process area
        * @param Wsrc : Pitch of process matrix, in element (uint8_t)
        * @param _illeagal_space_v32 : The length of idle area in the last vector during the data loading
        */
        static void _histgen2D_u8_caller(const uint8_t* src, uint64_t* histogram, const uint2 proc_dims, const uint32_t Wsrc);
    }
}



static void decx::bp::_histgen2D_u8_caller(const uint8_t* src, uint64_t* histogram, const uint2 proc_dims, const uint32_t Wsrc)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, decx::cpu::_get_permitted_concurrency());
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    const uint8_t _illeagal_space_v32 = (uint8_t)(proc_dims.x % 32);

    decx::PtrInfo<uint64_t> _hist_for_threads;
    if (decx::alloc::_host_virtual_page_malloc(&_hist_for_threads, t1D.total_thread * 256 * sizeof(uint64_t))) {
        return;
    }
    const uint8_t* loc_src = src;
    uint64_t *loc_hist = _hist_for_threads.ptr;

    for (int thread_id = 0; thread_id < t1D.total_thread - 1; ++thread_id) {
        t1D._async_thread[thread_id] = decx::cpu::register_task_default( decx::bp::CPUK::_histgen2D_u8,
            loc_src, loc_hist,
            make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, 32), f_mgr.frag_len), Wsrc,
            _illeagal_space_v32);

        loc_src += (Wsrc * f_mgr.frag_len);
        loc_hist += 256;
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::bp::CPUK::_histgen2D_u8,
        loc_src, loc_hist,
        make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, 32), _L), Wsrc,
        _illeagal_space_v32);

    t1D.__sync_all_threads();

    __m256i accu, recv;
    for (int i = 0; i < 256 / 4; ++i) {
        accu = _mm256_castpd_si256(_mm256_load_pd((double*)(_hist_for_threads.ptr + i * 4)));
        for (int threadId = 1; threadId < t1D.total_thread; ++threadId) {
            recv = _mm256_castpd_si256(_mm256_load_pd((double*)(_hist_for_threads.ptr + i * 4 + threadId * 256)));
            accu = _mm256_add_epi64(accu, recv);
        }
        _mm256_store_pd((double*)histogram + i * 4, _mm256_castsi256_pd(accu));
    }

    decx::alloc::_host_virtual_page_dealloc(&_hist_for_threads);
}


#endif