/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _IM2COL_GEMM_FP32_CUH_
#define _IM2COL_GEMM_FP32_CUH_

#include "../../../../../common/basic.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../../common/FMGR/fragment_arrangment.h"


#define _IM2COL_GEMM_FP32_BLOCK_X_ _CUDA_WARP_SIZE_
#define _IM2COL_GEMM_FP32_BLOCK_Y_ 8


namespace decx
{
namespace nn {
    namespace GPUK 
    {
        __global__ void cu_im2col_GEMM_fp32(const float4* im2col_buf, const float4* kernel,
            float4* dst, const uint32_t dpitch_dst_v1, const uint32_t wpitch_i2c_v1, 
            const uint32_t wpitch_dst_v1, const uint32_t _L_proc_v1, const uint2 conv2D_area);
    }
}
}


#endif
