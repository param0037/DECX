/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _IM2ROW_FP32_CUH_
#define _IM2ROW_FP32_CUH_

#include "../../../core/basic.h"



namespace decx
{
    namespace conv_I2R 
    {
        namespace GPUK {
            __global__
            /**
            * Wdst = channel * kernel._element_num, 64x
            * Hdst = src.element_num 16x
            * Each thread process 4 float4s each loop, 4 lines in dst matrix
            * @param thread_bound : the boundary of threads, .x : in float4, the width of src(the Tensor) / 4
            *                                                 .y : the height of src(the Tensor)
            * @param Wpitch_src : src_buf->Wpitch / 4, in float4
            * @param depth_iter : how many loops should have along the depth, dpitch --> float4 once
            * @param pitch_dst : the pitch of dst, which is equal to channel * kernel_width * kernel_height / 4, in float4
            * @param ker_size : the size of kernel, (pitch, height)
            */
            void cu_Im2Col_v128_fp32(const float4* src, float4* dst, const int2 thread_bound,
                const size_t Wpitch_src, const size_t pitch_dst, const int2 ker_size, const int depth);


            __global__
            void cu_Im2Col_v128_fp16(const float4* src, float4* dst, const int2 src_load_bound,
                const size_t Wpitch_src, const size_t pitch_dst, const int2 ker_size, const int depth);


            __global__
            /**
            * Wdst = channel * kernel._element_num, 64x
            * Hdst = src.element_num 16x
            * Each thread process 4 float4s each loop, 4 lines in dst matrix
            * @param thread_bound : the boundary of threads, .x : in float4, the width of src(the Tensor) / 4
            *                                                 .y : the height of src(the Tensor)
            * @param Wpitch_src : src_buf->Wpitch / 4, in float4
            * @param depth_iter : how many loops should have along the depth, dpitch --> float4 once
            * @param pitch_dst : the pitch of dst, which is equal to channel * kernel_width * kernel_height / 4, in float4
            * @param ker_size : the size of kernel, (pitch, height)
            */
            void cu_Im2Col_v128_stride_fp32(const float4* src, float4* dst, const uint2 strideXY, const int2 thread_bound,
                const size_t Wpitch_src, const size_t pitch_dst, const int2 ker_size, const int depth);


            __global__
            void cu_Im2Col_v128_stride_fp16(const float4* src, float4* dst, const uint2 strideXY, const int2 thread_bound,
                const size_t Wpitch_src, const size_t pitch_dst, const int2 ker_size, const int depth);
        }
    }
}

#endif