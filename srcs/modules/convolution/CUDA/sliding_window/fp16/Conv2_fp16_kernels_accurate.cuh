/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_FP16_KERNELS_ACCURATE_CUH_
#define _CONV2_FP16_KERNELS_ACCURATE_CUH_


#include "../../../../core/basic.h"
#include "../../../../classes/classes_util.h"
#include "../Conv2_kernel_defs.cuh"


__global__
void cu_hConv2_r8_exact_accu(float4* src,
    float4* dst,
    const uint         pitch_src,
    const uint         pitch_dst,
    const uint         total_ker_len,
    const uint         Wker);




__global__
void cu_hConv2_r8_exact_offset_accu(float4* src,
    float4* dst,
    const uint          pitch_src,
    const uint          pitch_dst,
    const uint          total_ker_len,
    const uint          Wker,
    const size_t        offset);




__global__
void cu_hConv2_r8_within_accu(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift);




__global__
void cu_hConv2_r8_within_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const int2             kernel_shift,
    const size_t           offset);




__global__
void cu_hConv2_r16_exact_accu(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker);




__global__
void cu_hConv2_r16_exact_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const size_t           offset);




__global__
void cu_hConv2_r16_within_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const int2             kernel_shift);




__global__
void cu_hConv2_r16_within_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const int2             kernel_shift,
    const size_t           offset);



__global__
void cu_hConv2_r816_exact_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker);





__global__
void cu_hConv2_r816_exact_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const size_t           offset);




__global__
void cu_hConv2_r816_within_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const int2             kernel_shift);




__global__
void cu_hConv2_r816_within_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const int2             kernel_shift,
    const size_t           offset);




__global__
void cu_hConv2_r168_exact_accu(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker);




__global__
void cu_hConv2_r168_exact_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const size_t           offset);



__global__
/**
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* �����ά����8����
* */
void cu_hConv2_r168_within_accu(float4* src,
    float4* dst,
    const uint              pitch_src,
    const uint              pitch_dst,
    const uint              total_ker_len,
    const uint              Wker,
    const int2              kernel_shift);





__global__
/**
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* �����ά����8����
* */
void cu_hConv2_r168_within_offset_accu(float4* src,
    float4* dst,
    const uint             pitch_src,
    const uint             pitch_dst,
    const uint             total_ker_len,
    const uint             Wker,
    const int2             kernel_shift,
    const size_t           offset);




#endif