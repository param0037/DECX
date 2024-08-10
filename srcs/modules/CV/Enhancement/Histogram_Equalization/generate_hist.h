/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GENERATE_HIST_H_
#define _GENERATE_HIST_H_


#include "../../../../common/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../../common/FMGR/fragment_arrangment.h"


namespace decx
{
    namespace vis {
        namespace CPUK 
        {
            /*
            * Accumulate the histogram of the __m256 (32 x uchar8)
            */
            _THREAD_CALL_ inline void
                _hist_accumulate_vec32(const decx::utils::simd::xmm256_reg _reg_uc32, uint32_t* _hist);

            /*
            * Accumulate the histogram of the __m256 (32 x uchar8) (locally, in any length)
            */
            _THREAD_CALL_ inline void
                _hist_accumulate_vec_any(const decx::utils::simd::xmm256_reg _reg_uc32, uint32_t* _hist, const uint16_t _L);


            _THREAD_FUNCTION_ void
                _generate_hist_uc8(const float* src, uint32_t *hist, const uint2 proc_dims, const uint32_t pitch);
        }
    }
}


#endif