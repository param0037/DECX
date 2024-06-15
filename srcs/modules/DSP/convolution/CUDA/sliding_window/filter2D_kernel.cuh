/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FILER2D_FP32_KERNEL_CUH_
#define _FILER2D_FP32_KERNEL_CUH_

#include "../../../../../core/basic.h"
#include "../../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../../core/utils/decx_cuda_math_functions.cuh"
#include "../cuda_filter2D_planner.cuh"


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
            const uint32_t pitchsrc_v4, const uint32_t pitchdst_v4, const uint3 kernel_dims, const uint2 conv_area);


        template <uint32_t _ext_w> __global__ 
        void cu_filter2D_BC_fp64(const double2* __restrict src, const double* __restrict kernel, double2* __restrict dst,
            const uint32_t pitchsrc_v4, const uint32_t pitchdst_v4, const uint3 kernel_dims, const uint2 conv_area);

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
