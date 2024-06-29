/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _CONV2_FP64_EXEC_H_
#define _CONV2_FP64_EXEC_H_


#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "conv2_fp64_K_loop_core.h"
#include "../../../conv_utils.h"

#define _BLOCKED_CONV2_FP64_H_ 4
#define _BLOCKED_CONV2_FP64_W_ 8


namespace decx
{
    namespace conv {
        namespace CPUK {
            // r1 ~ r2 (control by rwx)
            _THREAD_CALL_ void _conv2_r1_rect_fixed_fp64_ST(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst);


            _THREAD_CALL_ void _conv2_r1_rect_flex_fp64_ST(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst);


            _THREAD_CALL_ void _conv2_r2_rect_fixed_fp64_ST(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst);


            _THREAD_CALL_ void _conv2_r2_rect_flex_fp64_ST(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst);


            // rN
            _THREAD_CALL_ void _conv2_rN_rect_fixed_fp64_ST_rw2(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_CALL_ void _conv2_rN_rect_flex_fp64_ST_rw2(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_CALL_ void _conv2_rN_rect_fixed_fp64_ST_rw4(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_CALL_ void _conv2_rN_rect_flex_fp64_ST_rw4(double* __restrict src, double* kernel, double* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const uint _loop);
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
                void _conv2_rN_fp64_ST_rw2(double* src, double* kernel, double* dst, const uint2 proc_dim,
                    const uint2 ker_dims, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_FUNCTION_
                void _conv2_rN_fp64_ST_rw4(double* src, double* kernel, double* dst, const uint2 proc_dim,
                    const uint2 ker_dims, const uint Wsrc, const uint Wdst, const uint _loop);


            _THREAD_FUNCTION_
                void _conv2_r1_fp64_ST(double* src, double* kernel, double* dst, const uint2 proc_dim,
                    const uint2 ker_dims, const uint Wsrc, const uint Wdst);


            _THREAD_FUNCTION_
                void _conv2_r2_fp64_ST(double* src, double* kernel, double* dst, const uint2 proc_dim,
                    const uint2 ker_dims, const uint Wsrc, const uint Wdst);
        }
    }
}




namespace decx
{
    namespace conv {
        void _conv2_r1_r2_fp64_caller(double* src, double* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);



        void _conv2_rN_fp64_caller(double* src, double* kernel, double* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);
    }
}



namespace decx
{
    static void _conv2_fp64_organiser(double* src, double* kernel, double* dst,
        const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
        const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
}



void decx::_conv2_fp64_organiser(double*                      src,
                                 double*                      kernel, 
                                 double*                      dst, 
                                 const uint2                  proc_dim, 
                                 const uint2                  ker_dims,
                                 const uint                   Wsrc,
                                 const uint                   Wdst,
                                 decx::utils::_thr_1D*        t1D,
                                 decx::utils::frag_manager*   f_mgr)
{
    const uint half_kernel_w = ker_dims.x / 2;
    if (half_kernel_w < 3) {
        decx::conv::_conv2_r1_r2_fp64_caller(src, kernel, dst, proc_dim, ker_dims, Wsrc, Wdst, t1D, f_mgr);
    }
    else {
        const uint _loop = (uint)decx::utils::clamp_min<int>((((int)ker_dims.x / 2) - 3) / 2, 0);
        ushort reg_WL = (ushort)(ker_dims.x - 1 - 4 * (_loop + 1));
        decx::conv::_conv2_rN_fp64_caller(src, kernel, dst, proc_dim, ker_dims, Wsrc, Wdst, reg_WL, t1D, f_mgr, _loop);
    }
}


#endif