/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_FP64_BC_MK_EXEC_H_
#define _CONV2_FP64_BC_MK_EXEC_H_


#include "conv2_fp64_exec.h"
#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/array_ptr_info.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "../../../../../classes/classes_util.h"
#include "conv2_fp64_BC_SK_exec.h"


namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_ void _conv2_rN_BC_MK_fp32_ST_top_rw2(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_MK_fp32_ST_mid_rw2(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_MK_fp32_ST_bottom_rw2(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_MK_fp32_ST_top_rw4(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_MK_fp32_ST_mid_rw4(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_MK_fp32_ST_bottom_rw4(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2 proc_dim, const decx::_C2_MK64* conv2_mk_props);
        }
    }
}



namespace decx
{
    namespace conv {
        namespace CPUK {
            // r1, r4
            _THREAD_CALL_ void _conv2_r1_BC_MK_fp64_ST_top(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2  proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r1_BC_MK_fp64_ST_mid(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2  proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r1_BC_MK_fp64_ST_bottom(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2  proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r2_BC_MK_fp64_ST_top(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2  proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r2_BC_MK_fp64_ST_mid(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2  proc_dim, const decx::_C2_MK64* conv2_mk_props);


            _THREAD_CALL_ void _conv2_r2_BC_MK_fp64_ST_bottom(double* __restrict src, double* __restrict tmp_src, double* __restrict kernel,
                double* __restrict dst, const uint2  proc_dim, const decx::_C2_MK64* conv2_mk_props);
        }
    }
}



namespace decx
{
    namespace conv {
        void _conv2_rN_BC_MK_fp64_caller(double* src, double* tmp_src, double* kernel, double* dst,
            const uint2 proc_dim, decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_props);


        void _conv2_r1_r2_BC_MK_fp64_caller(double* src, double* tmp_src, double* kernel, double* dst,
            const uint2 proc_dim, decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_props);
    }
}



namespace decx
{
    namespace conv {
        static void _conv2_MK_BC_fp64_organiser(double* src, double* tmp_src, double* kernel, double* dst,
            const uint2 proc_dim, decx::_C2_MK64* conv2_props, decx::utils::_thr_1D* t1D);
    }
}




void decx::conv::_conv2_MK_BC_fp64_organiser(double*                        src,
                                       double*                        tmp_src,
                                       double*                        kernel, 
                                       double*                        dst,
                                       const uint2                    proc_dim, 
                                       decx::_C2_MK64*                conv2_props, 
                                       decx::utils::_thr_1D*          t1D)
{
    const uint half_kernel_w = conv2_props->ker_dims.x / 2;
    if (half_kernel_w < 3) {
        decx::conv::_conv2_r1_r2_BC_MK_fp64_caller(src, tmp_src, kernel, dst, proc_dim, t1D, conv2_props);
    }
    else {
        decx::conv::_conv2_rN_BC_MK_fp64_caller(src, tmp_src, kernel, dst, proc_dim, t1D, conv2_props);
    }
}



#endif