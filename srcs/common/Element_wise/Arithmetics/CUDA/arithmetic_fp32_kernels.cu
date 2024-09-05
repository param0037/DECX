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

#include "../arithmetic_kernels.h"
#include "arithmetic_kernel_defs.cuh"


namespace decx
{
namespace GPUK{
    __global__ void cu_add_fp32_kernel(const float4* __restrict, const float4* __restrict, float4* __restrict, const uint64_t);
    __global__ void cu_sub_fp32_kernel(const float4* __restrict, const float4* __restrict, float4* __restrict, const uint64_t);
    __global__ void cu_mul_fp32_kernel(const float4* __restrict, const float4* __restrict, float4* __restrict, const uint64_t);
    __global__ void cu_div_fp32_kernel(const float4* __restrict, const float4* __restrict, float4* __restrict, const uint64_t);
    __global__ void cu_min_fp32_kernel(const float4* __restrict, const float4* __restrict, float4* __restrict, const uint64_t);
    __global__ void cu_max_fp32_kernel(const float4* __restrict, const float4* __restrict, float4* __restrict, const uint64_t);

    __global__ void cu_addc_fp32_kernel(const float4* __restrict, const float, float4* __restrict, const uint64_t);
    __global__ void cu_subc_fp32_kernel(const float4* __restrict, const float, float4* __restrict, const uint64_t);
    __global__ void cu_mulc_fp32_kernel(const float4* __restrict, const float, float4* __restrict, const uint64_t);
    __global__ void cu_divc_fp32_kernel(const float4* __restrict, const float, float4* __restrict, const uint64_t);
    __global__ void cu_minc_fp32_kernel(const float4* __restrict, const float, float4* __restrict, const uint64_t);
    __global__ void cu_maxc_fp32_kernel(const float4* __restrict, const float, float4* __restrict, const uint64_t);



}
}


#define exec_fp32_VO(permitives) stg._vf.x = permitives(recv._vf.x);    \
                                 stg._vf.y = permitives(recv._vf.y);    \
                                 stg._vf.z = permitives(recv._vf.z);    \
                                 stg._vf.w = permitives(recv._vf.w);    \

#define exec_fp32_VVO(permitives) stg._vf.x = permitives(recv_A._vf.x, recv_B._vf.x);   \
                                 stg._vf.y = permitives(recv_A._vf.y, recv_B._vf.y);    \
                                 stg._vf.z = permitives(recv_A._vf.z, recv_B._vf.z);    \
                                 stg._vf.w = permitives(recv_A._vf.w, recv_B._vf.w);    \

#define exec_fp32_VCO(permitives) stg._vf.x = permitives(recv._vf.x, constant);   \
                                 stg._vf.y = permitives(recv._vf.y, constant);    \
                                 stg._vf.z = permitives(recv._vf.z, constant);    \
                                 stg._vf.w = permitives(recv._vf.w, constant);    \

// kernels
_CUDA_OP_1D_VVO_(cu_add_fp32_kernel, _vf, exec_fp32_VVO(__fadd_rn));
_CUDA_OP_1D_VVO_(cu_sub_fp32_kernel, _vf, exec_fp32_VVO(__fsub_rn));
_CUDA_OP_1D_VVO_(cu_mul_fp32_kernel, _vf, exec_fp32_VVO(__fmul_rn));
_CUDA_OP_1D_VVO_(cu_div_fp32_kernel, _vf, exec_fp32_VVO(__fdividef));
_CUDA_OP_1D_VVO_(cu_min_fp32_kernel, _vf, exec_fp32_VVO(min));
_CUDA_OP_1D_VVO_(cu_max_fp32_kernel, _vf, exec_fp32_VVO(max));

_CUDA_OP_1D_VCO_(cu_addc_fp32_kernel, _vf, float, exec_fp32_VCO(__fadd_rn));
_CUDA_OP_1D_VCO_(cu_subc_fp32_kernel, _vf, float, exec_fp32_VCO(__fsub_rn));
_CUDA_OP_1D_VCO_(cu_mulc_fp32_kernel, _vf, float, exec_fp32_VCO(__fmul_rn));
_CUDA_OP_1D_VCO_(cu_divc_fp32_kernel, _vf, float, exec_fp32_VCO(__fdividef));
_CUDA_OP_1D_VCO_(cu_minc_fp32_kernel, _vf, float, exec_fp32_VCO(min));
_CUDA_OP_1D_VCO_(cu_maxc_fp32_kernel, _vf, float, exec_fp32_VCO(max));

// kernel callers
_CUDA_OP_1D_VVO_CALLER_(_add_fp32_kernel, cu_add_fp32_kernel, float, _vf);
_CUDA_OP_1D_VVO_CALLER_(_sub_fp32_kernel, cu_sub_fp32_kernel, float, _vf);
_CUDA_OP_1D_VVO_CALLER_(_mul_fp32_kernel, cu_mul_fp32_kernel, float, _vf);
_CUDA_OP_1D_VVO_CALLER_(_div_fp32_kernel, cu_div_fp32_kernel, float, _vf);
_CUDA_OP_1D_VVO_CALLER_(_max_fp32_kernel, cu_max_fp32_kernel, float, _vf);
_CUDA_OP_1D_VVO_CALLER_(_min_fp32_kernel, cu_min_fp32_kernel, float, _vf);

_CUDA_OP_1D_VCO_CALLER_(_addc_fp32_kernel, cu_addc_fp32_kernel, float, float, _vf);
_CUDA_OP_1D_VCO_CALLER_(_subc_fp32_kernel, cu_subc_fp32_kernel, float, float, _vf);
_CUDA_OP_1D_VCO_CALLER_(_mulc_fp32_kernel, cu_mulc_fp32_kernel, float, float, _vf);
_CUDA_OP_1D_VCO_CALLER_(_divc_fp32_kernel, cu_divc_fp32_kernel, float, float, _vf);
_CUDA_OP_1D_VCO_CALLER_(_maxc_fp32_kernel, cu_maxc_fp32_kernel, float, float, _vf);
_CUDA_OP_1D_VCO_CALLER_(_minc_fp32_kernel, cu_minc_fp32_kernel, float, float, _vf);


// void decx::GPUK::
// _add_fp32_kernel(const float* __restrict A, 
//                  const float* __restrict B, 
//                  float* __restrict dst, 
//                  const uint64_t proc_len_v,
//                  const uint32_t block, 
//                  const uint32_t grid)
// {
    
// }