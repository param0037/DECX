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

#ifndef _CUDA_GATHER_KERNELS_H_
#define _CUDA_GATHER_KERNELS_H_


#include "../../../basic.h"
#include "../../../../modules/core/cudaStream_management/cudaStream_queue.h"


namespace decx
{
namespace GPUK{
    void vgather2D_fp32(cudaTextureObject_t tex, const float2* map, float* dst,
        const uint2 src_dims_v1, const uint2 proc_dims, const uint32_t pitchmap_v1, const uint32_t pitchdst_v,
        dim3 block, dim3 grid, decx::cuda_stream* S);


    void vgather2D_uint8(cudaTextureObject_t tex, const float2* map, uint8_t* dst,
        const uint2 src_dims_v1, const uint2 proc_dims, const uint32_t pitchmap_v1, const uint32_t pitchdst_v,
        dim3 block, dim3 grid, decx::cuda_stream* S);


    void vgather2D_uchar4(cudaTextureObject_t tex, const float2* map, uchar4* dst,
        const uint2 src_dims_v1, const uint2 proc_dims, const uint32_t pitchmap_v1, const uint32_t pitchdst_v,
        dim3 block, dim3 grid, decx::cuda_stream* S);

    
    template <typename _data_type>
    using cuda_vgather_kernel = void(cudaTextureObject_t, const float2*, _data_type*,
        const uint2, const uint2, const uint32_t, const uint32_t, dim3, dim3, decx::cuda_stream*);
}
}


#endif

