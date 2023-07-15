/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DISTRIBUTIONS_EXEC_H_
#define _DISTRIBUTIONS_EXEC_H_


#include "../../../core/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../core/thread_management/thread_arrange.h"


namespace decx
{
    namespace gen {
        namespace CPUK {
            _THREAD_FUNCTION_ void _gaussian2D_fp32(float* _target, const float expect, const float diveation,
                const uint2 proc_dims, const uint32_t Wsrc, const float2 clip_range, const __m256 blend_var,
                const uint32_t resolution);
        }

        /**
        * @param proc_dims : ~.x -> width of process area, in element (uint8_t)
        * ~.y -> height of process area
        * @param Wsrc : Pitch of process matrix, in element (uint8_t)
        * @param _illeagal_space_v32 : The length of idle area in the last vector during the data loading
        */
        static void _gaussian2D_fp32_caller(float* target, const float expect, const float diveation,
            const uint2 proc_dims, const uint32_t Wsrc, const float2 clip_range, const uint32_t resolution);
    }
}


static void decx::gen::_gaussian2D_fp32_caller(float* target, const float expect, const float diveation,
    const uint2 proc_dims, const uint32_t Wsrc, const float2 clip_range, const uint32_t resolution)
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims.y, decx::cpu::_get_permitted_concurrency());
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    const uint8_t occupied_length = (uint8_t)(proc_dims.x % 8);

    decx::utils::simd::xmm256_reg blend_var;
    blend_var._vf = _mm256_setzero_ps();
    for (uint8_t i = occupied_length; i < 8; ++i) {
#ifdef __GNUC__
        ((int*)&blend_var._vi)[i] = 0xFFFFFFFF;
#endif
#ifdef _MSC_VER
        blend_var._vi.m256i_i32[i] = 0xFFFFFFFF;
#endif
    }

    float* loc_target = target;
    for (int thread_id = 0; thread_id < t1D.total_thread - 1; ++thread_id) {
        t1D._async_thread[thread_id] = decx::cpu::register_task_default( decx::gen::CPUK::_gaussian2D_fp32,
            loc_target, expect, diveation, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, 8), f_mgr.frag_len),
            Wsrc, clip_range, blend_var._vf, resolution);

        loc_target += (Wsrc * f_mgr.frag_len);
    }
    const uint32_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] = decx::cpu::register_task_default( decx::gen::CPUK::_gaussian2D_fp32,
        loc_target, expect, diveation, make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, 8), _L),
        Wsrc, clip_range, blend_var._vf, resolution);

    t1D.__sync_all_threads();
}


#endif