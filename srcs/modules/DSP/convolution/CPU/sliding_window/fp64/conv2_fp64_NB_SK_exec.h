/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_FP64_NB_SK_H_
#define _CONV2_FP64_NB_SK_H_


#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../conv_utils.h"
#include "conv2_fp64_exec.h"


#if !defined(_fmgr)
#define _fmgr decx::utils::frag_manager
#endif


namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_
                void _conv2_rN_fp64_ST_rw2_unconfigured(double* src, double* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const _fmgr* f_mgrH, const _fmgr* f_mgrW, const uint _loop);


            _THREAD_CALL_
                void _conv2_rN_fp64_ST_rw4_unconfigured(double* src, double* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const _fmgr* f_mgrH, const _fmgr* f_mgrW, const uint _loop);


            _THREAD_CALL_
                void _conv2_r1_fp64_ST_unconfigured(double* src, double* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const _fmgr* f_mgrH, const _fmgr* f_mgrW);


            _THREAD_CALL_
                void _conv2_r2_fp64_ST_unconfigured(double* src, double* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const _fmgr* f_mgrH, const _fmgr* f_mgrW);
        }
    }
}



namespace decx
{
    namespace conv {
        namespace CPUK {
            /*
            * This function is suitable only when 3 __m256 are needed to cover all the data on one row
            * which indicates that half_kerdim.x -> 5
            *
            * @param proc_dim : .x -> in _m256, the width of proccess area of signle thread (on dst matrix)
            *                   .y -> the height of proccess area of signle thread (on dst matrix)
            * @param ker_dim : .x -> the width of kernel (in float); .y -> the height of kernel
            * @param reg_WL : the leftover on width. ( = (half_ker_dims.x * 2 + 8) - 2 * 8)
            * @param Wsrc : the pitch of src matrix, in __m256
            */
            _THREAD_FUNCTION_
                void _conv2_rN_fp64_NB_SK_rw2(double* src, double* kernel, double* dst, const uint2 proc_dim, const size_t page_size_src,
                    const size_t page_size_dst, const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_FUNCTION_
                void _conv2_rN_fp64_NB_SK_rw4(double* src, double* kernel, double* dst, const uint2 proc_dim, const size_t page_size_src,
                    const size_t page_size_dst, const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_FUNCTION_
                void _conv2_r1_fp64_NB_SK(double* src, double* kernel, double* dst, const uint2 proc_dim, const size_t page_size_src,
                    const size_t page_size_dst, const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst);


            _THREAD_FUNCTION_
                void _conv2_r2_fp64_NB_SK(double* src, double* kernel, double* dst, const uint2 proc_dim, const size_t page_size_src,
                    const size_t page_size_dst, const uint2 ker_dims, const uint channel_size, const uint Wsrc, const uint Wdst);
        }
    }
}



namespace decx
{
    namespace conv {
        void _conv2_rN_NB_SK_fp64_caller(double* src, double* kernel, double* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_mk_props);


        void _conv2_r1_r2_NB_SK_fp64_caller(double* src, double* kernel, double* dst, const uint2 proc_dim,
            decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_mk_props);
    }
}




namespace decx
{
    namespace conv {
        static void _conv2_NB_SK_fp64_organiser(double* src, double* kernel, double* dst,
            const uint2 proc_dim, decx::utils::_thr_1D* t1D, decx::_C2_MK64* conv2_mk_props);
    }
}


void decx::conv::_conv2_NB_SK_fp64_organiser(double*                    src,
                                       double*                    kernel, 
                                       double*                    dst,
                                       const uint2                proc_dim,
                                       decx::utils::_thr_1D*      t1D,
                                       decx::_C2_MK64*            conv2_mk_props)
{
    const uint half_kernel_w = conv2_mk_props->ker_dims.x / 2;
    if (half_kernel_w < 3) {
        decx::conv::_conv2_r1_r2_NB_SK_fp64_caller(src, kernel, dst, proc_dim, t1D, conv2_mk_props);
    }
    else {
        decx::conv::_conv2_rN_NB_SK_fp64_caller(src, kernel, dst, proc_dim, t1D, conv2_mk_props);
    }
}


#endif