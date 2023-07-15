/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _SCONV2_KERNEL_CALLERS_H_
#define _SCONV2_KERNEL_CALLERS_H_


#include "../../../../core/basic.h"
#include "Conv2_fp32_kernels.cuh"
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
        static inline void conv2_fp32_kernel_r8x8(const float4* src,    const float* kernel,        float4* dst,
                                                   const uint2 kernel_shift,             const uint2 src_dims,
                                                   const uint2 dst_dims,                 const uint2 ker_dims,
                                                   decx::cuda_stream* S,                 const uint _const_mem_offset = 0)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);

            const dim3 grids(decx::utils::ceil<uint>(dst_dims.y, conv2_bld), 
                                 decx::utils::ceil<uint>(dst_dims.x, conv2_bld));
        
            decx::conv::GPUK::cu_sConv2_r8_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                src, kernel, dst,
                src_dims.x,
                dst_dims.x,
                ker_dims.x * ker_dims.y,
                ker_dims.x,
                kernel_shift,
                dst_dims);
        }


        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp32_kernel_r16x16(const float4* src, const float* kernel, float4* dst,
                                                    const uint2 kernel_shift,             const uint2 src_dims,
                                                    const uint2 dst_dims,                 const uint2 ker_dims,
                                                    decx::cuda_stream* S,                 const uint _const_mem_offset = 0)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.y, conv2_bld), 
                             decx::utils::ceil<uint>(dst_dims.x, conv2_bld));
        
            decx::conv::GPUK::cu_sConv2_r16_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                src, kernel, dst,
                src_dims.x,
                dst_dims.x,
                ker_dims.x * ker_dims.y,
                ker_dims.x,
                kernel_shift,
                dst_dims);
        }


        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp32_kernel_r8x16(const float4* src, const float* kernel, float4* dst,
                                                    const uint2 kernel_shift,             const uint2 src_dims,
                                                    const uint2 dst_dims,                 const uint2 ker_dims,
                                                    decx::cuda_stream* S,                 const uint _const_mem_offset = 0)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.y, conv2_bld), 
                             decx::utils::ceil<uint>(dst_dims.x, conv2_bld));

            decx::conv::GPUK::cu_sConv2_r816_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                src,kernel, dst,
                src_dims.x,
                dst_dims.x,
                ker_dims.x * ker_dims.y,
                ker_dims.x,
                kernel_shift,
                dst_dims);
        }


        /**\
        * The conv2 kernel, this function complete one single convolution, regardless
        * of borders
        * @param Dsrc : device ptr of source matrix
        * @param Ddst : device ptr of destination matrix
        * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
        * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
        */
        static inline void conv2_fp32_kernel_r16x8(const float4* src, const float* kernel, float4* dst,
                                              const uint2 kernel_shift,             const uint2 src_dims,
                                              const uint2 dst_dims,                 const uint2 ker_dims,
                                              decx::cuda_stream* S,                 const uint _const_mem_offset = 0)    noexcept
        {
            const dim3 threads(conv2_bld, conv2_bld);
            const dim3 grids(decx::utils::ceil<uint>(dst_dims.y, conv2_bld), 
                             decx::utils::ceil<uint>(dst_dims.x, conv2_bld));

            decx::conv::GPUK::cu_sConv2_r168_within << <grids, threads, 0, S->get_raw_stream_ref() >> > (
                src, kernel, dst,
                src_dims.x,
                dst_dims.x,
                ker_dims.x * ker_dims.y,
                ker_dims.x,
                kernel_shift,
                dst_dims);
        }
    }
}


#endif