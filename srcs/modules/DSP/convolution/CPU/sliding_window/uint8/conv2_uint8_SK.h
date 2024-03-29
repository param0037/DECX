/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_FP32_NB_SK_EXEC_H_
#define _CONV2_FP32_NB_SK_EXEC_H_

#include "conv2_uint8_exec.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "conv2_uint8_BC_SK_exec.h"


namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_ void _conv2_rN_NB_SK_fp32_ST(float* src, float* kernel,
                float* dst, const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            // r1, r14
            _THREAD_CALL_ void _conv2_r1_r4_NB_SK_fp32_ST(float* src, float* kernel,
                float* dst, const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst);
        }
    }
}



namespace decx
{
    namespace conv {
        void _conv2_rN_NB_SK_fp32_caller(float* src, float* kernel, float* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK32* conv2_mk_props);


        void _conv2_r1_r4_NB_SK_fp32_caller(float* src, float* kernel, float* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK32* conv2_mk_props);
    }
}


#endif