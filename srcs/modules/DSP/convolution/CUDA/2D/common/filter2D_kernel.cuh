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


#ifndef _FILER2D_FP32_KERNEL_CUH_
#define _FILER2D_FP32_KERNEL_CUH_

#include "../../../../../../common/basic.h"
#include "../../../../../../common/CUSV/decx_cuda_vectypes_ops.cuh"
#include "../../../../../../common/CUSV/decx_cuda_math_functions.cuh"
#include "cuda_filter2D_planner.cuh"


namespace decx
{
namespace dsp {
    namespace GPUK 
    {
        // ------------------------------------------------- FP32 -------------------------------------------------

        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_NB_fp32(const float4* __restrict src, const float* __restrict kernel, float4* __restrict dst,
            const uint32_t pitchsrc_v4, const uint32_t pitchdst_v4, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_fp32(const float4* __restrict src, const float* __restrict kernel, float4* __restrict dst,
            const uint32_t pitchsrc_v4, const uint32_t pitchdst_v4, const uint3 kernel_dims, const uint2 conv_area);

        // ------------------------------------------------- FP64 -------------------------------------------------

        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_NB_fp64(const double2* __restrict src, const double* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v2, const uint32_t pitchdst_v2, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_fp64(const double2* __restrict src, const double* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v2, const uint32_t pitchdst_v2, const uint3 kernel_dims, const uint2 conv_area);

        // ------------------------------------------------- COMPLEX_F32 -------------------------------------------

        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_NB_cplxf(const double2* __restrict src, const de::CPf* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v2, const uint32_t pitchdst_v2, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_cplxf(const double2* __restrict src, const de::CPf* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v2, const uint32_t pitchdst_v2, const uint3 kernel_dims, const uint2 conv_area);

        // ------------------------------------------------- COMPLEX_F64 -------------------------------------------

        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_NB_cplxd(const double2* __restrict src, const de::CPd* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v1, const uint32_t pitchdst_v1, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_cplxd(const double2* __restrict src, const de::CPd* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v1, const uint32_t pitchdst_v1, const uint3 kernel_dims, const uint2 conv_area);


        // ------------------------------------------------- UINT8 -------------------------------------------------

        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_NB_u8_Kfp32_fp32(const double* __restrict src, const float* __restrict kernel, float4* __restrict dst,
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v8, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_u8_Kfp32_fp32(const double* __restrict src, const float* __restrict kernel, float4* __restrict dst,
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v8, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_NB_u8_Kfp32_u8(const double* __restrict src, const float* __restrict kernel, double* __restrict dst,
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v8, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_u8_Kfp32_u8(const double* __restrict src, const float* __restrict kernel, double* __restrict dst,
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v8, const uint3 kernel_dims, const uint2 conv_area);
    }
}
}


#define _CU_FILTER2D_SPEC_(funcname, type1, type2, type3, templ_val)                                                \
    template __global__ void decx::dsp::GPUK::funcname<templ_val>(const type1* __restrict, const type2* __restrict, \
    type3* __restrict, const uint32_t, const uint32_t, const uint3, const uint2)                                    \


#endif
