/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_FP16_KERNELS_CUH_
#define _CONV2_FP16_KERNELS_CUH_


#include "../../../../core/basic.h"
#include "../../../../classes/classes_util.h"
#include "../Conv2_kernel_defs.cuh"


__global__
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
void cu_hConv2_r8_exact(float4* src,
    float4* dst,
    const uint         pitch_src,
    const uint         pitch_dst,
    const uint         total_ker_len,
    const uint         Wker);


__global__
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
* -------------------------------
* |                                |        8 halfs
* |         -----------------        |
* |         |                 |        |
* |    apron|     constant     |        |
* |         |                 |        |
* |         -----------------        |
* |                                |
* -------------------------------
*/
void cu_hConv2_r8_within(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift);



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 128 + 16 * 2)*(16 + 16 * 2) 即shmem half[48][160]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
* 
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
* 
*     \160 floats (80 floats)    16 floats (20 float4s = (16 + 4) float4s)
* -------------------------------
* |                                |        16 halfs
* |         -----------------        |
* |         |                 |        |
* |    apron|     constant     |        |        48 halfs  => __shared__ float src_frag[48][160]
* |         |                 |        |
* |         -----------------        |
* |                                |
* -------------------------------
*/
void cu_hConv2_r16_exact(float4*            src, 
                         float4*            dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker);



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_hConv2_r16_within(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift);



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 16 * 2)*(16 + 8 * 2) 即shmem half[32][160]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dimension of Dsrc is +16 * 2 = 32floats (+8 float4s if considered float4) on width
* The dimension of Dsrc is +8 * 2 = 16floats on height
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
*
*     \160 halfs (80floats)     16 halfs (20 float4s = (16 + 4) float4s)
* -------------------------------
* |         -----------------        |        8 halfs
* |         |                 |        |
* |    apron|     constant     |        |        32 floats  => __shared__ float src_frag[32][160]
* |         |                 |        |
* |         -----------------        |
* -------------------------------
*/
void cu_hConv2_r816_exact(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker);



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_hConv2_r816_within(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift);




__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x8个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 8 + 8 * 2)*(16 + 8 * 2) 即shmem half[48][144]
* So the alignments should be x64(16 * 4) in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 16 * 2 = 32 on all directions(if float4 is consider horizentally, then +8 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度刚好都等于8
*
*     \144 halfs (72 floats)    8 halfs (16 + 2)float4s
* -----------------------
* |           apron        |        16 halfs
* |     ----------------    |
* |     |                |    |
* |     |     constant    |    |        48 halfs  => __shared__ float src_frag[48][144]
* |     |                |    |
* |     ----------------    |
* |                        |
* -----------------------
*/
void cu_hConv2_r168_exact(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker);



__global__
/**
* The radius of convolutional kernel = 8，每个线程处理1x4个数据(one float4)，一个块16x16个线程，
* 即一个块需要的共享内大小为(16 * 4 + 8 * 2)*(16 + 8 * 2) 即shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* 核心域：64 x 16(floats), 光环域：8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* 卷积核维度在8以内
* */
void cu_hConv2_r168_within(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift);



__global__
void cu_hConv2_r8_exact_offset(float4* src,
    float4* dst,
    const uint        pitch_src,
    const uint        pitch_dst,
    const uint        total_ker_len,
    const uint        Wker,
    const size_t     offset);



__global__
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
* -------------------------------
* |                                |        8 halfs
* |         -----------------        |
* |         |                 |        |
* |    apron|     constant     |        |        
* |         |                 |        |
* |         -----------------        |
* |                                |
* -------------------------------
*/
void cu_hConv2_r8_within_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const int2            kernel_shift,
                                const size_t        offset);



__global__
void cu_hConv2_r16_exact_offset(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const size_t        offset);



__global__
void cu_hConv2_r16_within_offset(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift,
    const size_t        offset);



__global__
void cu_hConv2_r816_exact_offset(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const size_t        offset);



__global__
void cu_hConv2_r816_within_offset(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift,
    const size_t            offset);




__global__
void cu_hConv2_r168_exact_offset(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const size_t        offset);



__global__
void cu_hConv2_r168_within_offset(float4* src,
    float4* dst,
    const uint            pitch_src,
    const uint            pitch_dst,
    const uint            total_ker_len,
    const uint            Wker,
    const int2            kernel_shift,
    const size_t            offset);



#endif