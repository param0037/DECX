/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _IM2COL_GEMM_FP32_CUH_
#define _IM2COL_GEMM_FP32_CUH_

#include "../../../../core/basic.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"


#define _IM2COL_GEMM_FP32_BLOCK_X_ _CUDA_WARP_SIZE_
#define _IM2COL_GEMM_FP32_BLOCK_Y_ 8


namespace decx
{
namespace nn {
    namespace GPUK {
        __global__ void cu_im2col_GEMM_fp32(const float4* im2col_buf, const float4* kernel,
            float4* dst, const uint32_t dpitch_dst_v1, const uint32_t wpitch_i2c_v1, 
            const uint32_t wpitch_dst_v1, const uint32_t _L_proc_v1, const uint2 conv2D_area);
    }
}
}


#endif
