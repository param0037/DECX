/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _HCONV2_KERNEL_CALLERS_H_
#define _HCONV2_KERNEL_CALLERS_H_


#include "../../../../core/basic.h"
#include "../fp16/Conv2_fp16_kernels.cuh"
#include "../fp16/Conv2_fp16_kernels_accurate.cuh"
#include "../fp32/Conv2_fp32_kernels.cuh"
#include "../../../conv_utils.h"


namespace decx
{
    /**\
    * The conv2 kernel, this function complete one single convolution, regardless
    * of borders
    * @param Dsrc : device ptr of source matrix
    * @param Ddst : device ptr of destination matrix
    * @param src_dim : ~.x : pitch_src (having considered float4); ~.y : Hsrc, in float
    * @param dst_dim : ~.x : pitch_dst (having considered float4); ~.y : Hdst, in float
    */
    static inline void hconv2_kernel_within8x8(float4*                  Dsrc,
                                               float4*                  Ddst,
                                               const int2               kernel_shift,
                                               const int2               src_dim,
                                               const int2               dst_dim,
                                               const int2               ker_dim,
                                               decx::cuda_stream*       S,
                                               const int                flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r8_within << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r8_within_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        default:
            break;
        }
    }

    static inline void hconv2_kernel_within8x8_offset(float4*                   Dsrc,
                                                      float4*                   Ddst,
                                                      const int2                kernel_shift,
                                                      const int2                src_dim,
                                                      const int2                dst_dim,
                                                      const int2                ker_dim,
                                                      const size_t              offset,
                                                      decx::cuda_stream*        S,
                                                      const int                 flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r8_within_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r8_within_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
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
    static inline void hconv2_kernel_exact8x8(float4*                   Dsrc,
                                              float4*                   Ddst,
                                              const int2                src_dim,
                                              const int2                dst_dim,
                                              const int2                ker_dim,
                                              decx::cuda_stream*        S,
                                              const int                 flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r8_exact << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r8_exact_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        default:
            break;
        }
    }

    static inline void hconv2_kernel_exact8x8_offset(float4* Dsrc,
        float4* Ddst,
        const int2 src_dim,
        const int2 dst_dim,
        const int2 ker_dim,
        const size_t offset,
        decx::cuda_stream* S,
        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r8_exact_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r8_exact_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
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
    static inline void hconv2_kernel_exact16x16(float4*                  Dsrc,
                                                float4*                  Ddst,
                                                const int2               src_dim,
                                                const int2               dst_dim,
                                                const int2               ker_dim,
                                                decx::cuda_stream*       S,
                                                const int                flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r16_exact << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r16_exact_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        default:
            break;
        }
    }

    static inline void hconv2_kernel_exact16x16_offset(float4* Dsrc,
        float4* Ddst,
        const int2 src_dim,
        const int2 dst_dim,
        const int2 ker_dim,
        const size_t offset,
        decx::cuda_stream* S,
        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r16_exact_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r16_exact_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
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
    static inline void hconv2_kernel_within16x16(float4*                    Dsrc,
                                                 float4*                    Ddst,
                                                 const int2                 kernel_shift,
                                                 const int2                 src_dim,
                                                 const int2                 dst_dim,
                                                 const int2                 ker_dim,
                                                 decx::cuda_stream*         S,
                                                 const int                  flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r16_within << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r16_within_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        default:
            break;
        }
    }

    static inline void hconv2_kernel_within16x16_offset(float4* Dsrc,
                                                        float4* Ddst,
                                                        const int2 kernel_shift,
                                                        const int2 src_dim,
                                                        const int2 dst_dim,
                                                        const int2 ker_dim,
                                                        const size_t offset,
                                                        decx::cuda_stream* S,
                                                        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r16_within_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r16_within_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
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
    static inline void hconv2_kernel_exact8x16(float4*                  Dsrc,
                                               float4*                  Ddst,
                                               const int2               src_dim,
                                               const int2               dst_dim,
                                               const int2               ker_dim,
                                               decx::cuda_stream*       S,
                                               const int                flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r816_exact << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r816_exact_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        default:
            break;
        }
    }

    static inline void hconv2_kernel_exact8x16_offset(float4* Dsrc,
        float4* Ddst,
        const int2 src_dim,
        const int2 dst_dim,
        const int2 ker_dim,
        const size_t offset,
        decx::cuda_stream* S,
        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r816_exact_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r816_exact_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
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
    static inline void hconv2_kernel_within8x16(float4*                     Dsrc,
                                                float4*                     Ddst,
                                                const int2                  kernel_shift,
                                                const int2                  src_dim,
                                                const int2                  dst_dim,
                                                const int2                  ker_dim,
                                                decx::cuda_stream*          S,
                                                const int                   flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r816_within << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r816_within_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        default:
            break;
        }
    }

    static inline void hconv2_kernel_within8x16_offset(float4* Dsrc,
        float4* Ddst,
        const int2 kernel_shift,
        const int2 src_dim,
        const int2 dst_dim,
        const int2 ker_dim,
        const size_t offset,
        decx::cuda_stream* S,
        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r816_within_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r816_within_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
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
    static inline void hconv2_kernel_exact16x8(float4*              Dsrc,
                                               float4*              Ddst,
                                               const int2           src_dim,
                                               const int2           dst_dim,
                                               const int2           ker_dim,
                                               decx::cuda_stream*   S,
                                               const int            flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));

        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r168_exact << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r168_exact_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x);
            break;

        default:
            break;
        }
        
    }

    static inline void hconv2_kernel_exact16x8_offset(float4* Dsrc,
        float4* Ddst,
        const int2 src_dim,
        const int2 dst_dim,
        const int2 ker_dim,
        const size_t offset,
        decx::cuda_stream* S,
        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r168_exact_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r168_exact_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, offset);
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
    static inline void hconv2_kernel_within16x8(float4*                     Dsrc,
                                                float4*                     Ddst,
                                                const int2                  kernel_shift,
                                                const int2                  src_dim,
                                                const int2                  dst_dim,
                                                const int2                  ker_dim,
                                                decx::cuda_stream*          S,
                                                const int                   flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r168_within << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r168_within_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift);
            break;
        default:
            break;
        }
    }

    static inline void hconv2_kernel_within16x8_offset(float4* Dsrc,
        float4* Ddst,
        const int2 kernel_shift,
        const int2 src_dim,
        const int2 dst_dim,
        const int2 ker_dim,
        const size_t offset,
        decx::cuda_stream* S,
        const int flag)    noexcept
    {
        const dim3 threads(conv2_bld, conv2_bld);
        const dim3 blocks(dst_dim.y / (conv2_bld), dst_dim.x / (conv2_bld));
        switch (flag)
        {
        case decx::conv_property::half_conv_ordinary:
            cu_hConv2_r168_within_offset << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
            break;

        case decx::conv_property::half_conv_accurate:
            cu_hConv2_r168_within_offset_accu << <blocks, threads, 0, S->get_raw_stream_ref() >> > (
                Dsrc, Ddst, src_dim.x, dst_dim.x, ker_dim.x * ker_dim.y, ker_dim.x, kernel_shift, offset);
            break;
        default:
            break;
        }
    }

}


#endif