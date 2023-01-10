/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV2_FP32_NB_MK_EXEC_H_
#define _CONV2_FP32_NB_MK_EXEC_H_


#include "conv2_fp32_exec.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/MatrixArray.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../conv_utils.h"
#include "conv2_fp32_BC_SK_exec.h"


namespace decx
{
    namespace conv {
        namespace CPUK {

            _THREAD_CALL_ void _conv2_rN_NB_MK_fp32_ST(float* src, float* kernel, float* dst,
                const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst, const size_t page_size_ker,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            _THREAD_CALL_ void _conv2_r1_r4_NB_MK_fp32_ST(float* src, float* kernel, float* dst,
                const uint2 proc_dim, const size_t page_size_src, const size_t page_size_dst, const size_t page_size_ker,
                const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst);
        }
    }
}




namespace decx
{
    namespace conv {
        void _conv2_rN_NB_MK_fp32_caller(float* src, float* kernel, float* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK32* conv2_mk_props);



        void _conv2_r1_r4_NB_MK_fp32_caller(float* src, float* kernel, float* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK32* conv2_mk_props);
    }
}




namespace decx
{
    namespace conv {
        static void _conv2_r8_NB_MK_fp32_organiser(float* src, float* kernel, float* dst,
            const uint2 proc_dim, decx::utils::_thr_1D* t1D,
            decx::_C2_MK32* conv2_mk_props);
    }
}



void decx::conv::_conv2_r8_NB_MK_fp32_organiser(float*                        src,
                                          float*                        kernel, 
                                          float*                        dst,
                                          const uint2                   proc_dim,
                                          decx::utils::_thr_1D*         t1D,
                                          decx::_C2_MK32*               conv2_mk_props)
{
    const uint half_kernel_w = conv2_mk_props->ker_dims.x / 2;
    if (half_kernel_w < 5) {
        decx::conv::_conv2_r1_r4_NB_MK_fp32_caller(src, kernel, dst, proc_dim, t1D, conv2_mk_props);
    }
    else {
        decx::conv::_conv2_rN_NB_MK_fp32_caller(src, kernel, dst, proc_dim, t1D, conv2_mk_props);
    }
}



#endif