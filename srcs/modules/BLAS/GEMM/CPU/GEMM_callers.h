/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_CALLERS_H_
#define _GEMM_CALLERS_H_

#include "../GEMM_utils.h"
#include "cpu_GEMM_config.h"

namespace decx
{
    namespace blas {
        template <bool _ABC>
        void GEMM_fp32_caller(const float* A, const float* B, float* dst, const decx::_matrix_layout* layout_A,
            const decx::_matrix_layout* layout_dst, const uint2 _arranged_B_dims, const decx::utils::frag_manager *f_mgrH,
            const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::_thr_2D* t1D, const float* C = NULL);
    }
}




#endif