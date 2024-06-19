/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_B_ARRANGE_H_
#define _MATRIX_B_ARRANGE_H_

#include "../../../classes/Matrix.h"
#include "../GEMM_utils.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/thread_management/thread_pool.h"


namespace decx
{
    namespace blas {
        void matrix_B_arrange_fp32(const float* src, float* dst, const uint32_t pitchsrc_v1,
            const uint32_t pitchdst_v8, const decx::utils::frag_manager* _fmgr_WH, decx::utils::_thr_2D* t2D);
    }
}

#endif
