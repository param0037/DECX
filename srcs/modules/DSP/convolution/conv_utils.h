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


#ifndef _CONV_UTILS_H_
#define _CONV_UTILS_H_

#ifdef _DECX_DSP_CPU_
#include "../../core/vector_defines.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/thread_management/thread_arrange.h"
#endif


#include "../../classes/Matrix.h"
#include "../../classes/MatrixArray.h"
#include "../../classes/Tensor.h"
#include "../../classes/TensorArray.h"

#ifdef _DECX_DSP_CUDA_
#include "../../classes/GPU_Matrix.h"
#include "../../classes/GPU_MatrixArray.h"
#include "../../classes/GPU_Tensor.h"
#include "../../classes/GPU_TensorArray.h"
#include "../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../core/cudaStream_management/cudaStream_queue.h"
#endif

#ifdef _DECX_DSP_CPU_
#include "../../BLAS/basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#endif


namespace decx {
    enum conv_property
    {
        de_conv_no_compensate = 0,
        de_conv_zero_compensate = 1,

        half_conv_ordinary = 2,
        half_conv_accurate = 3
    };
}


#ifdef _DECX_DSP_CUDA_

namespace decx
{
    namespace conv 
    {
        struct _matrix_configs
        {
            uint _width, _pitch, _height, _matrix_num;
            void* _ptr;
            void** _ptr_array;

            void gen_matrix_configs(decx::_Matrix* _host_mat);


            void gen_matrix_configs(decx::_GPU_Matrix* _host_mat);


            void gen_matrix_configs(decx::_MatrixArray* _host_mat);


            void gen_matrix_configs(decx::_GPU_MatrixArray* _host_mat);


            _matrix_configs();
        };
    }
}




#define N_conv_once 1024 * 1024

// the total bytes (IN BYTE) allowed for kernel to restore in constant memory
#define kernel_in_CM 512


#define conv2_bld 16
#define conv2_tps 4

#define bounded_kernel_R8  8
#define bounded_kernel_R16 16
#define bounded_kernel_R64 64


#endif

#if defined(_DECX_BLAS_CPU_) || defined(_DECX_DSP_CPU_) || defined(_DECX_CV_CPU_)
namespace decx
{
    typedef struct _Conv2_MK_Props_fp32
    {
        uint2 ker_dims;
        uint Wdst;
        uint Wsrc;

        ushort reg_WL;
        uint _loop;

        decx::utils::frag_manager* f_mgr; 
        size_t page_size_dst;
        size_t page_size_src;
        size_t page_size_ker;
        uint channel_size;

        _Conv2_MK_Props_fp32(const uint2                 _ker_dims, 
                             const uint                  _W_original_src, 
                             const uint                  _W_tmp_src, 
                             const size_t                _page_size, 
                             const uint                  _channel_size,
                             decx::utils::frag_manager*  _f_mgr,
                             const size_t                _page_size_src = 0,      // set only when non-border conv2d occurs
                             const size_t                _page_size_ker = 0);       // set only when multi-kernel conv2d occurs
    }_C2_MK32;


    typedef struct _Conv2_MK_Props_fp64
    {
        uint2 ker_dims;
        uint Wdst;
        uint Wsrc;

        ushort reg_WL;
        uint _loop;

        decx::utils::frag_manager* f_mgr; 
        size_t page_size_dst;
        size_t page_size_src;
        size_t page_size_ker;
        uint channel_size;

        _Conv2_MK_Props_fp64(const uint2                 _ker_dims,
                             const uint                  _W_original_src, 
                             const uint                  _W_tmp_src, 
                             const size_t                _page_size, 
                             const uint                  _channel_size,
                             decx::utils::frag_manager*  _f_mgr,
                             const size_t                _page_size_src = 0,      // set only when non-border conv2d occurs
                             const size_t                _page_size_ker = 0);       // set only when multi-kernel conv2d occurs

    }_C2_MK64;
}


namespace decx
{
    static void _thread_dispatch_for_conv2(decx::utils::frag_manager** f_mgr,
        const size_t tot, const uint thread_num, const uint N, const uint Wproc);


    static void _thread_dispatch_for_conv2_fp64(decx::utils::frag_manager** f_mgr,
        const size_t tot, const uint thread_num, const uint N, const uint Wproc);
}


#define _CONV2_THR_DIST_CRIT_R1R4_ 4096
#define _CONV2_THR_DIST_CRIT_R5R8_ 1024
#define _CONV2_THR_DIST_CRIT_R9R12_ 256

#define _CONV2_THR_DIST_CRIT_FP64_R1R4_ 2048
#define _CONV2_THR_DIST_CRIT_FP64_R5R8_ 512
#define _CONV2_THR_DIST_CRIT_FP64_R9R12_ 128


static void decx::_thread_dispatch_for_conv2(decx::utils::frag_manager** f_mgr, const size_t tot,
    const uint thread_num, const uint N, const uint Wproc)
{
    decx::utils::frag_manager* f_mgr_N = new decx::utils::frag_manager;
    decx::utils::frag_manager_gen_Nx(f_mgr_N, tot, thread_num, N);

    if (f_mgr_N->is_left) {
        size_t _exceeded = (size_t)(f_mgr_N->frag_left_over - f_mgr_N->frag_len) * (size_t)Wproc;
        if (_exceeded > _CONV2_THR_DIST_CRIT_R5R8_) {
            decx::utils::frag_manager_gen(f_mgr_N, tot, thread_num);
            *f_mgr = f_mgr_N;
            return;
        }
    }
    *f_mgr = f_mgr_N;
}



static void decx::_thread_dispatch_for_conv2_fp64(decx::utils::frag_manager** f_mgr, const size_t tot,
    const uint thread_num, const uint N, const uint Wproc)
{
    decx::utils::frag_manager* f_mgr_N = new decx::utils::frag_manager;
    decx::utils::frag_manager_gen_Nx(f_mgr_N, tot, thread_num, N);

    if (f_mgr_N->is_left) {
        size_t _exceeded = (size_t)(f_mgr_N->frag_left_over - f_mgr_N->frag_len) * (size_t)Wproc;
        if (_exceeded > _CONV2_THR_DIST_CRIT_FP64_R5R8_) {
            decx::utils::frag_manager_gen(f_mgr_N, tot, thread_num);
            *f_mgr = f_mgr_N;
            return;
        }
    }
    *f_mgr = f_mgr_N;
}
#endif


#endif