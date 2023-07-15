/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _IM2ROW_EQ_GEMM_FP32_H_
#define _IM2ROW_EQ_GEMM_FP32_H_


#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/utils/intrinsics_ops.h"
#include "../../../../core/utils/fragment_arrangment.h"


#define _im2row_eqMM_frag_size_ 64


namespace decx {
    namespace conv_I2R {
        namespace CPUK {
            _THREAD_FUNCTION_ void
                _im2row_eq_GEMM_fp32_ST(const float* I2C_buf, const float* kernel, float* dst, const uint WI2C,
                    const size_t proc_WH, const uint dst_dpitch, const uint kernel_TN);
        }

        void _im2row_eq_GEMM_caller_fp32(const float* I2C_buf, const float* kernel, float* dst, const uint WI2C,
            const size_t proc_WH, const uint dst_dpitch, const uint kernel_TN, decx::utils::_thread_arrange_1D* t1D);
    }
}


#endif