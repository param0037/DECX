/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_UINT8_KERNEL_CALLERS_H_
#define _CONV2_UINT8_KERNEL_CALLERS_H_


#include "../../../../core/basic.h"
#include "Conv2_uint8_kernels.cuh"
#include "../../../conv_utils.h"


namespace decx
{
    namespace conv
    {
        static void conv2_uc8_kfp32_kernels_caller(const float4* src, const float* kernel, float2* dst, const uint pitchsrc,
            const uint pitchdst, const uint2 ker_dims, const uint2 kernel_shift, const uint2 dst_dim, decx::cuda_stream* S)
        {
            const dim3 threads(conv2_bld, conv2_bld);
            const dim3 grids(decx::utils::ceil<uint>(dst_dim.y, conv2_bld), 
                             decx::utils::ceil<uint>(dst_dim.x, conv2_bld));

            if (ker_dims.y / 2 > bounded_kernel_R8) {
                decx::conv::_conv2_LKH LKH = decx::conv::make_LKH(ker_dims.y, 17);

                decx::conv::GPUK::cu_Conv2_r64x8_uc8_kfp32_LKH << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, kernel, dst, pitchsrc, pitchdst, ker_dims.x, kernel_shift.y, decx::conv::make_LKH(ker_dims.y, 17), dst_dim);
            }
            else {
                decx::conv::GPUK::cu_Conv2_r64x8_uc8_kfp32 << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, kernel, dst, pitchsrc, pitchdst, ker_dims.x * ker_dims.y, ker_dims.x, kernel_shift, dst_dim);
            }
        }


        static void conv2_uc8_fp32_kfp32_kernels_caller(const float4* src, const float* kernel, float4* dst, const uint pitchsrc,
            const uint pitchdst, const uint2 ker_dims, const uint2 kernel_shift, const uint2 dst_dim, decx::cuda_stream* S)
        {
            const dim3 threads(conv2_bld, conv2_bld);
            const dim3 grids(decx::utils::ceil<uint>(dst_dim.y, conv2_bld),
                decx::utils::ceil<uint>(dst_dim.x, conv2_bld));

            if (ker_dims.y / 2 > bounded_kernel_R8) {
                decx::conv::_conv2_LKH LKH = decx::conv::make_LKH(ker_dims.y, 17);

                decx::conv::GPUK::cu_Conv2_r64x8_uc8_fp32_kfp32_LKH << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, kernel, dst, pitchsrc, pitchdst, ker_dims.x, kernel_shift.y, decx::conv::make_LKH(ker_dims.y, 17), dst_dim);
            }
            else {
                decx::conv::GPUK::cu_Conv2_r64x8_uc8_fp32_kfp32 << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, kernel, dst, pitchsrc, pitchdst, ker_dims.x * ker_dims.y, ker_dims.x, kernel_shift, dst_dim);
            }
        }
    }
}


#endif