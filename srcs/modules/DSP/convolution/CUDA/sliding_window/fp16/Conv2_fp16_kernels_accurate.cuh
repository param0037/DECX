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


#include "../../../../../core/basic.h"
#include "../../../../../classes/classes_util.h"
#include "../Conv2_kernel_defs.cuh"
#include "Conv2_fp16_kernels.cuh"


namespace decx {
    namespace conv {
        namespace GPUK {

            __global__
                void cu_hConv2_r8_within_accu(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint            pitch_src,
                    const uint            pitch_dst,
                    const uint            total_ker_len,
                    const uint            Wker,
                    const uint2            kernel_shift,
                    const uint2           dst_dims);





            __global__
                void cu_hConv2_r168_within_accu(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint              pitch_src,
                    const uint              pitch_dst,
                    const uint              total_ker_len,
                    const uint              Wker,
                    const uint2              kernel_shift,
                    const uint2             dst_dims);





            __global__
                void cu_hConv2_r816_within_accu(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint             pitch_src,
                    const uint             pitch_dst,
                    const uint             total_ker_len,
                    const uint             Wker,
                    const uint2             kernel_shift,
                    const uint2            dst_dims);





            __global__
                void cu_hConv2_r16_within_accu(const float4* src,
                    const __half* kernel,
                    float4* dst,
                    const uint             pitch_src,
                    const uint             pitch_dst,
                    const uint             total_ker_len,
                    const uint             Wker,
                    const uint2             kernel_shift,
                    const uint2            dst_dims);

        }
    }
}

#endif