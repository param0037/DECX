/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_CUH_
#define _CONV2_CUH_

#include "../../../../../core/basic.h"
#include "../../../../../classes/classes_util.h"
#include "../Conv2_kernel_defs.cuh"


/**
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions
*
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* ������ά�ȸպö�����8
*
*             \80 floats     8 floats
* ----------------------------------
* |                                |        8 halfs
* |        -----------------       |
* |       |                 |      |
* |  apron|     constant    |      |        32 floats  => __shared__ float src_frag[32][80]
* |       |                 |      |
* |        -----------------       |
* |                                |
* ----------------------------------
*/


/**
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem float[48][96]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
*
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* ������ά�ȸպö�����8
*
*            \96 floats     16 floats
* -----------------------------------
* |                                 |        16 floats
* |         -----------------       |
* |         |               |       |
* |    apron|     constant  |       |        48 floats  => __shared__ float src_frag[32][80]
* |         |               |       |
* |         -----------------       |
* |                                 |
* -----------------------------------
*/


namespace decx
{
namespace conv
{
    namespace GPUK{
    __global__
    void cu_sConv2_r8_within(const float4* __restrict src, const float* __restrict kernel,
                                float4* __restrict dst, const uint32_t pitch_src,
                                const uint32_t pitch_dst, const uint2 kernel_dims, 
                            const uint2 kernel_shift, const uint2 dst_dims);



    __global__
    void cu_sConv2_r16_within(const float4* __restrict src, const float* __restrict kernel,
        float4* __restrict dst, const uint32_t pitch_src,
        const uint32_t pitch_dst, const uint2 kernel_dims,
        const uint2 kernel_shift, const uint2 dst_dims);




    __global__
    void cu_sConv2_r816_within(const float4* __restrict src, const float* __restrict kernel,
        float4* __restrict dst, const uint32_t pitch_src,
        const uint32_t pitch_dst, const uint2 kernel_dims,
        const uint2 kernel_shift, const uint2 dst_dims);



    __global__
    void cu_sConv2_r168_within(const float4* __restrict src, const float* __restrict kernel,
        float4* __restrict dst, const uint32_t pitch_src,
        const uint32_t pitch_dst, const uint2 kernel_dims,
        const uint2 kernel_shift, const uint2 dst_dims);
    }
}
}





#endif