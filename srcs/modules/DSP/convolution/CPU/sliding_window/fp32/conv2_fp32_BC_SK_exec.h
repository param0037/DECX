/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_FP32_BC_SK_EXEC_H_
#define _CONV2_FP32_BC_SK_EXEC_H_

#include "conv2_fp32_exec.h"
#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/array_ptr_info.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "../../../../../classes/classes_util.h"


#if !defined(_fmgr)
#define _fmgr decx::utils::frag_manager
#endif

namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_
                void _conv2_rN_fp32_ST_unconfigured(float* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const ushort reg_WL, const _fmgr* f_mgrH, const _fmgr* f_mgrW, const uint _loop);


            _THREAD_CALL_
                void _conv2_r1_r4_fp32_ST_unconfigured(float* src, float* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const _fmgr* f_mgrH, const _fmgr* f_mgrW);
        }
    }
}



namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_ void _conv2_rN_BC_SK_fp32_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_SK_fp32_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_SK_fp32_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props);


            // r1, r4
            _THREAD_CALL_ void _conv2_r1_r4_BC_SK_fp32_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2  proc_dim, const decx::_C2_MK32* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r1_r4_BC_SK_fp32_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2  proc_dim, const decx::_C2_MK32* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r1_r4_BC_SK_fp32_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2  proc_dim, const decx::_C2_MK32* conv2_mk_props);
        }
    }
}




namespace decx
{
    namespace conv {
        void _conv2_rN_BC_SK_fp32_caller(float* src, float* tmp_src, float* kernel, float* dst,
            const uint2 proc_dim, decx::utils::_thr_1D* t1D, decx::_C2_MK32* conv2_props);


        void _conv2_r1_r4_BC_SK_fp32_caller(float* src, float* tmp_src, float* kernel, float* dst,
            const uint2 proc_dim, decx::utils::_thr_1D* t1D, decx::_C2_MK32* conv2_props);
    }
}




namespace decx
{
    namespace conv {
        static void _conv2_SK_BC_fp32_organiser(float* src, float* tmp_src, float* kernel, float* dst,
            const uint2 proc_dim, decx::_C2_MK32* conv2_props, decx::utils::_thr_1D* t1D);
    }
}




void decx::conv::_conv2_SK_BC_fp32_organiser(float*                        src,
                                       float*                        tmp_src,
                                       float*                        kernel, 
                                       float*                        dst,
                                       const uint2                   proc_dim, 
                                       decx::_C2_MK32*               conv2_props, 
                                       decx::utils::_thr_1D*         t1D)
{
    const uint half_kernel_w = conv2_props->ker_dims.x / 2;
    if (half_kernel_w < 5) {
        decx::conv::_conv2_r1_r4_BC_SK_fp32_caller(src, tmp_src, kernel, dst, proc_dim, t1D, conv2_props);
    }
    else {
        decx::conv::_conv2_rN_BC_SK_fp32_caller(src, tmp_src, kernel, dst, proc_dim, t1D, conv2_props);
    }
}



#endif