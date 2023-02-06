/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _IM2ROW_FP32_CUH_
#define _IM2ROW_FP32_CUH_

#include "../../../core/basic.h"



#define store_to_shmem_L_vec4 {                                                                            \
    frag[threadIdx.x][4 * threadIdx.y] = reg_0[0];                                                        \
    frag[threadIdx.x][4 * threadIdx.y + 1] = reg_0[1];                                                    \
    frag[threadIdx.x][4 * threadIdx.y + 2] = reg_0[2];                                                    \
    frag[threadIdx.x][4 * threadIdx.y + 3] = reg_0[3];                                                    \
    frag[16 + threadIdx.x][4 * threadIdx.y] = reg_1[0];                                                    \
    frag[16 + threadIdx.x][4 * threadIdx.y + 1] = reg_1[1];                                                \
    frag[16 + threadIdx.x][4 * threadIdx.y + 2] = reg_1[2];                                                \
    frag[16 + threadIdx.x][4 * threadIdx.y + 3] = reg_1[3];                                                \
}                                                                                                        \



#define store_to_shmem_R_vec4 {                                                                            \
    frag[threadIdx.x][64 + 4 * threadIdx.y] = reg_0[0];                                                    \
    frag[threadIdx.x][65 + 4 * threadIdx.y] = reg_0[1];                                                    \
    frag[threadIdx.x][66 + 4 * threadIdx.y] = reg_0[2];                                                    \
    frag[threadIdx.x][67 + 4 * threadIdx.y] = reg_0[3];                                                    \
    frag[16 + threadIdx.x][64 + 4 * threadIdx.y] = reg_1[0];                                            \
    frag[16 + threadIdx.x][65 + 4 * threadIdx.y] = reg_1[1];                                            \
    frag[16 + threadIdx.x][66 + 4 * threadIdx.y] = reg_1[2];                                            \
    frag[16 + threadIdx.x][67 + 4 * threadIdx.y] = reg_1[3];                                            \
}                                                



__device__ __inline__
void reg_left_shift_float4_4(float4* src)
{
    src[0] = src[1];
    src[1] = src[2];
    src[2] = src[3];
}


/*
* 因为要适应不同的边界模式，因此需要拷贝，那么就可以不用考虑线程的边界问题了, 但拷贝到dst还是要注意边界问题
*/


/**
* Block(16, 16), the radius of preset border is 8
*
*                     80 float4s     
*            --------------------------
*            |         apron              |        8 floats
*            |      --------------     |
*            |      |               |     |
*            |      | constant   |     |        32 floats  => __shared__ float src_frag[32][80]
*            |      |               |     |
*            |      --------------     |
*            |                         |
*            --------------------------
*/

namespace decx
{
    namespace conv_I2R {
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
            void cu_Im2Row_v128_r8_within(float4*                       src,
                                      float4*                       dst,
                                      const int2                    kernel_shift,
                                      const int2                    thread_bound,
                                      const size_t                  Wpitch_src,
                                      const size_t                  pitch_dst,
                                      const int2                    ker_size,
                                      const int                     depth);



/**
* Block(16, 16), the radius of preset border is 8
*
*                 66 float4s
*                                     1 floats
*            ------------------ ->apron 
*            | -------------- |
*            | |               | |
*            | |  constant  | |        18 floats  => __shared__ float src_frag[32][80]
*            | |               | |
*            | -------------- |
*            ------------------
*/


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
            void cu_sIm2Row_r1_exact(float4*                       src,
                                      float4*                       dst,
                                      const int2                    thread_bound,
                                      const size_t                  Wpitch_src,
                                      const size_t                  pitch_dst,
                                      const int2                    ker_size,
                                      const int                     depth);



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
            void cu_sIm2Row_r2_exact(float4*                       src,
                                      float4*                       dst,
                                      const int2                    thread_bound,
                                      const size_t                  Wpitch_src,
                                      const size_t                  pitch_dst,
                                      const int2                    ker_size,
                                      const int                     depth);
        }
    }
}

#endif