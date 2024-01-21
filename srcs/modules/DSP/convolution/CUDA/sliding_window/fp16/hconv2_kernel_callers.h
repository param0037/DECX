/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _HCONV2_KERNEL_CALLERS_H_
#define _HCONV2_KERNEL_CALLERS_H_


#include "../../../../../core/basic.h"
#include "../fp16/Conv2_fp16_kernels.cuh"
#include "../fp16/Conv2_fp16_kernels_accurate.cuh"
#include "../fp32/Conv2_fp32_kernels.cuh"
#include "../../../conv_utils.h"


namespace decx
{
    namespace conv{
        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp16_kernel_r8x8(const float4* src, const de::Half* kernel, float4* dst,
                                                   const uint2 kernel_shift,             const uint2 src_dims,
                                                   const uint2 dst_dims,                 const uint2 ker_dims,
                                                   decx::cuda_stream* S,                const int _accu_flag)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.x, conv2_bld),
                             decx::utils::ceil<uint>(dst_dims.y, conv2_bld));

            switch (_accu_flag)
            {
            case decx::conv_property::half_conv_ordinary:
                decx::conv::GPUK::cu_hConv2_r8_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst, 
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            case decx::conv_property::half_conv_accurate:
                decx::conv::GPUK::cu_hConv2_r8_within_accu << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            default:
                break;
            }
        }



        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp16_kernel_r16x16(const float4* src, const de::Half* kernel, float4* dst,
                                                     const uint2 kernel_shift,             const uint2 src_dims,
                                                     const uint2 dst_dims,                 const uint2 ker_dims,
                                                     decx::cuda_stream* S,                const int _accu_flag)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
        
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.x, conv2_bld),
                             decx::utils::ceil<uint>(dst_dims.y, conv2_bld));

            switch (_accu_flag)
            {
            case decx::conv_property::half_conv_ordinary:
                decx::conv::GPUK::cu_hConv2_r16_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            case decx::conv_property::half_conv_accurate:
                decx::conv::GPUK::cu_hConv2_r16_within_accu << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            default:
                break;
            }
        }



        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp16_kernel_r8x16(const float4* src, const de::Half* kernel, float4* dst,
                                                    const uint2 kernel_shift,             const uint2 src_dims,
                                                    const uint2 dst_dims,                 const uint2 ker_dims,
                                                    decx::cuda_stream* S,                const int _accu_flag)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
        
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.x, conv2_bld),
                             decx::utils::ceil<uint>(dst_dims.y, conv2_bld));

            switch (_accu_flag)
            {
            case decx::conv_property::half_conv_ordinary:
                decx::conv::GPUK::cu_hConv2_r816_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            case decx::conv_property::half_conv_accurate:
                decx::conv::GPUK::cu_hConv2_r816_within_accu << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            default:
                break;
            }
        }



        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp16_kernel_r16x8(const float4* src, const de::Half* kernel, float4* dst,
                                                    const uint2 kernel_shift,             const uint2 src_dims,
                                                    const uint2 dst_dims,                 const uint2 ker_dims,
                                                    decx::cuda_stream* S,                const int _accu_flag)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
        
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.x, conv2_bld),
                             decx::utils::ceil<uint>(dst_dims.y, conv2_bld));

            switch (_accu_flag)
            {
            case decx::conv_property::half_conv_ordinary:
                decx::conv::GPUK::cu_hConv2_r168_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            case decx::conv_property::half_conv_accurate:
                decx::conv::GPUK::cu_hConv2_r168_within_accu << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                    src, (__half*)kernel, dst,
                    src_dims.x, dst_dims.x,
                    ker_dims, kernel_shift, dst_dims);
                break;

            default:
                break;
            }
        }
    }
}


#endif