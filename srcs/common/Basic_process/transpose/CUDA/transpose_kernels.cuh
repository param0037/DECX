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


#ifndef _TRANSPOSE_KERNELS_CUH_
#define _TRANSPOSE_KERNELS_CUH_


#include "../../../basic.h"
#include "../../../CUSV/decx_cuda_vectypes_ops.cuh"
#include "../../../CUSV/decx_cuda_math_functions.cuh"

#include "../../../../modules/core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../modules/core/cudaStream_management/cudaStream_queue.h"


namespace decx
{
namespace blas
{
     void transpose2D_b8(const double2* src, double2* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);

     void transpose2D_b16(const double2* src, double2* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);


#ifdef _DECX_DSP_CUDA_
     void transpose2D_b8_for_FFT(const double2* src, double2* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);


     void transpose2D_b16_for_FFT(const double2* src, double2* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);
#endif


     void transpose2D_b4(const float2* src, float2* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);


     void transpose2D_b2(const float4* src, float4* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);


     void transpose2D_b1(const uint32_t* src, uint32_t* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);
}
}


namespace decx
{
namespace blas
{
    void transpose2D_b4_dense(const float* src, float* dst, const uint2 proc_dims_dst,
        const uint32_t pitchsrc, const uint32_t pitchdst, decx::cuda_stream* S);
}
}


#define TRANSPOSE_MAT4X4(_loc_transp, tmp)              \
{                                                       \
SWAP(_loc_transp[0].y, _loc_transp[1].x, tmp);          \
SWAP(_loc_transp[0].z, _loc_transp[2].x, tmp);          \
SWAP(_loc_transp[0].w, _loc_transp[3].x, tmp);          \
                                                        \
SWAP(_loc_transp[1].z, _loc_transp[2].y, tmp);          \
                                                        \
SWAP(_loc_transp[1].w, _loc_transp[3].y, tmp);          \
SWAP(_loc_transp[2].w, _loc_transp[3].z, tmp);          \
}




#define TRANSPOSE_MAT8x8(_loc_transp, tmp)                                                  \
{/*row 0, col[1, 7]*/                                                                       \
SWAP(*(((__half*)&_loc_transp[0]) + 1), *(((__half*)&_loc_transp[1])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 2), *(((__half*)&_loc_transp[2])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 3), *(((__half*)&_loc_transp[3])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 4), *(((__half*)&_loc_transp[4])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 5), *(((__half*)&_loc_transp[5])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 6), *(((__half*)&_loc_transp[6])), tmp);                \
SWAP(*(((__half*)&_loc_transp[0]) + 7), *(((__half*)&_loc_transp[7])), tmp);                \
/*row1, col[2, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[1]) + 2), *(((__half*)&_loc_transp[2]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 3), *(((__half*)&_loc_transp[3]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 4), *(((__half*)&_loc_transp[4]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 5), *(((__half*)&_loc_transp[5]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 6), *(((__half*)&_loc_transp[6]) + 1), tmp);            \
SWAP(*(((__half*)&_loc_transp[1]) + 7), *(((__half*)&_loc_transp[7]) + 1), tmp);            \
/*row2, col[3, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[2]) + 3), *(((__half*)&_loc_transp[3]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 4), *(((__half*)&_loc_transp[4]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 5), *(((__half*)&_loc_transp[5]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 6), *(((__half*)&_loc_transp[6]) + 2), tmp);            \
SWAP(*(((__half*)&_loc_transp[2]) + 7), *(((__half*)&_loc_transp[7]) + 2), tmp);            \
/*row3, col[4, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[3]) + 4), *(((__half*)&_loc_transp[4]) + 3), tmp);            \
SWAP(*(((__half*)&_loc_transp[3]) + 5), *(((__half*)&_loc_transp[5]) + 3), tmp);            \
SWAP(*(((__half*)&_loc_transp[3]) + 6), *(((__half*)&_loc_transp[6]) + 3), tmp);            \
SWAP(*(((__half*)&_loc_transp[3]) + 7), *(((__half*)&_loc_transp[7]) + 3), tmp);            \
/*row4, col[5, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[4]) + 5), *(((__half*)&_loc_transp[5]) + 4), tmp);            \
SWAP(*(((__half*)&_loc_transp[4]) + 6), *(((__half*)&_loc_transp[6]) + 4), tmp);            \
SWAP(*(((__half*)&_loc_transp[4]) + 7), *(((__half*)&_loc_transp[7]) + 4), tmp);            \
/*row5, col[6, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[5]) + 6), *(((__half*)&_loc_transp[6]) + 5), tmp);            \
SWAP(*(((__half*)&_loc_transp[5]) + 7), *(((__half*)&_loc_transp[7]) + 5), tmp);            \
/*row6, col[7, 7]*/                                                                         \
SWAP(*(((__half*)&_loc_transp[6]) + 7), *(((__half*)&_loc_transp[7]) + 6), tmp);            \
}


#endif