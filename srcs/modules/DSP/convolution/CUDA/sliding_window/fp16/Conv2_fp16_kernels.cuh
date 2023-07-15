/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_FP16_KERNELS_CUH_
#define _CONV2_FP16_KERNELS_CUH_


#include "../../../../../core/basic.h"
#include "../../../../../classes/classes_util.h"
#include "../Conv2_kernel_defs.cuh"


/**
* The radius of convolutional kernel = 8，每个线程处理1x8个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 8 * 2)*(16 + 8 * 2) 即shmem half[32][144] -> float[32][72]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
*
*         \144halfs(72 floats)     8 halfs
* ----------------------------------
* |                                |        8 halfs
* |        -----------------       |
* |       |                 |      |
* |  apron|     constant    |      |
* |       |                 |      |
* |        -----------------       |
* |                                |
* ----------------------------------
*/


namespace decx {
    namespace conv {
        namespace GPUK {


            __global__
                void cu_hConv2_r8_within(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint            pitch_src,
                    const uint            pitch_dst,
                    const uint            total_ker_len,
                    const uint            Wker,
                    const uint2            kernel_shift,
                    const uint2           dst_dims);


            __global__
                void cu_hConv2_r168_within(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint            pitch_src,
                    const uint            pitch_dst,
                    const uint            total_ker_len,
                    const uint            Wker,
                    const uint2            kernel_shift,
                    const uint2           dst_dims);




            __global__
                void cu_hConv2_r816_within(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint            pitch_src,
                    const uint            pitch_dst,
                    const uint            total_ker_len,
                    const uint            Wker,
                    const uint2            kernel_shift,
                    const uint2           dst_dims);


            __global__
                void cu_hConv2_r16_within(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint            pitch_src,
                    const uint            pitch_dst,
                    const uint            total_ker_len,
                    const uint            Wker,
                    const uint2            kernel_shift,
                    const uint2           dst_dims);


        }
    }
}

#endif