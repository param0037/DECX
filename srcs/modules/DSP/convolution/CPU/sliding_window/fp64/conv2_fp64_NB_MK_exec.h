/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_FP64_NB_MK_EXEC_H_
#define _CONV2_FP64_NB_MK_EXEC_H_


#include "conv2_fp64_exec.h"
#include "../../../../../classes/Matrix.h"
#include "../../../../../classes/MatrixArray.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "../../../conv_utils.h"
#include "conv2_fp64_BC_SK_exec.h"


namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_ void _conv2_rN_NB_MK_fp64_ST_rw2(double* src, double* kernel, double* dst,
                const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst, const size_t page_size_ker,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_CALL_ void _conv2_rN_NB_MK_fp64_ST_rw4(double* src, double* kernel, double* dst,
                const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst, const size_t page_size_ker,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_CALL_ void _conv2_r1_NB_MK_fp64_ST(double* src, double* kernel, double* dst,
                const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst, const size_t page_size_ker,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst);


            _THREAD_CALL_ void _conv2_r2_NB_MK_fp64_ST(double* src, double* kernel, double* dst,
                const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst, const size_t page_size_ker,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst);
        }
    }
}




namespace decx
{
    namespace conv {
        void _conv2_rN_NB_MK_fp64_caller(double* src, double* kernel, double* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_mk_props);



        void _conv2_r1_r2_NB_MK_fp64_caller(double* src, double* kernel, double* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_mk_props);
    }
}




namespace decx
{
    namespace conv{
        static void _conv2_NB_MK_fp64_organiser(double* src, double* kernel, double* dst,
                                     const uint2 proc_dim, decx::utils::_thr_1D* t1D,
                                     decx::_C2_MK64* conv2_mk_props);
    }
}



void decx::conv::_conv2_NB_MK_fp64_organiser(double*                        src,
                                       double*                        kernel, 
                                       double*                        dst,
                                       const uint2                    proc_dim,
                                       decx::utils::_thr_1D*          t1D,
                                       decx::_C2_MK64*                conv2_mk_props)
{
    const uint half_kernel_w = conv2_mk_props->ker_dims.x / 2;
    if (half_kernel_w < 3) {
        decx::conv::_conv2_r1_r2_NB_MK_fp64_caller(src, kernel, dst, proc_dim, t1D, conv2_mk_props);
    }
    else {
        decx::conv::_conv2_rN_NB_MK_fp64_caller(src, kernel, dst, proc_dim, t1D, conv2_mk_props);
    }
}



#endif