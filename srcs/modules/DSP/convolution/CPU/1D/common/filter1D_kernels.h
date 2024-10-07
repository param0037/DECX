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

#ifndef _FILTER1D_KERNELS_H_
#define _FILTER1D_KERNELS_H_

#include "../../../../../../common/basic.h"
#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../../common/FMGR/fragment_arrangment.h"
#include "../../../../../core/thread_management/thread_arrange.h"


namespace decx
{
namespace dsp{
namespace CPUK{
    _THREAD_FUNCTION_ void conv1_fp32_kernel(const float* __restrict src, const float* kernel, float* __restrict dst,
        const uint32_t kernel_len, const uint32_t proc_len_v1);

    
    // Constant padding (zero padding)
    _THREAD_FUNCTION_ void conv1_BC_fp32_kernel(const float* __restrict src, const float* kernel, float* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1, 
        const uint32_t pitchdst_v1, const uint32_t start_rwo_id, const uint32_t Hsrc);


    // Reflective padding
    _THREAD_FUNCTION_ void conv1_BR_fp32_kernel(const float* __restrict src, const float* kernel, float* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1, const uint32_t start_rwo_id, const uint32_t Hsrc);


    typedef void conv1_B_kernel_fp32(const float*, const float*, float*, const uint2, const decx::utils::_blocking2D_fmgrs*, const uint32_t, 
        const uint32_t, const uint32_t, const uint32_t, const uint32_t);
}

namespace CPUK 
{
    _THREAD_FUNCTION_ void conv1_fp64_kernel(const double* __restrict src, const double* kernel, double* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1);


    // Constant padding (zero padding)
    _THREAD_FUNCTION_ void conv1_BC_fp64_kernel(const double* __restrict src, const double* kernel, double* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1, const uint32_t start_rwo_id, const uint32_t Hsrc);


    // Reflective padding
    _THREAD_FUNCTION_ void conv1_BR_fp64_kernel(const double* __restrict src, const double* kernel, double* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1, const uint32_t start_rwo_id, const uint32_t Hsrc);


    _THREAD_FUNCTION_ void conv1_cplxf_kernel(const double* __restrict src, const double* kernel, double* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1);


    // Constant padding (zero padding)
    _THREAD_FUNCTION_ void conv1_BC_cplxf_kernel(const double* __restrict src, const double* kernel, double* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1, const uint32_t start_rwo_id, const uint32_t Hsrc);


    // Reflective padding
    _THREAD_FUNCTION_ void conv1_BR_cplxf_kernel(const double* __restrict src, const double* kernel, double* __restrict dst,
        const uint2 kernel_dim, const decx::utils::_blocking2D_fmgrs* _block_conf, const uint32_t pitchsrc_v1, const uint32_t pitchkernel_v1,
        const uint32_t pitchdst_v1, const uint32_t start_rwo_id, const uint32_t Hsrc);


    typedef void conv1_NB_kernel_64b(const double*, const double*, double*, const uint2, const decx::utils::_blocking2D_fmgrs*, const uint32_t, 
        const uint32_t, const uint32_t);


    typedef void conv1_B_kernel_64b(const double*, const double*, double*, const uint2, const decx::utils::_blocking2D_fmgrs*, const uint32_t,
        const uint32_t, const uint32_t, const uint32_t, const uint32_t);
}
}
}


#endif