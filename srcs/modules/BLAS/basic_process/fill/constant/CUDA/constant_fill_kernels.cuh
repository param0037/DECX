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


#ifndef _CONSTANT_FILL_KERNELS_CUH_
#define _CONSTANT_FILL_KERNELS_CUH_


#include "../../../../../core/basic.h"
#include "../../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../../core/configs/config.h"


namespace decx
{
    namespace bp {
        namespace GPUK {
            /**
            * @param len : length of the array to be filled, in vec4
            */
            __global__ void
                cu_fill1D_constant_v128_b32(float4* src, const float4 val, const size_t len);


            /**
            * @param len : length of the array to be filled, in vec4
            */
            __global__ void
                cu_fill1D_constant_v128_b32_end(float4* src, const float4 val, const float4 _end_val, const size_t len);
        }

        void cu_fill1D_constant_v128_b32_caller(float* src, const float val, const size_t fill_len, decx::cuda_stream* S);


        void cu_fill1D_constant_v128_b64_caller(double* src, const double val, const size_t fill_len, decx::cuda_stream* S);
    }
}



namespace decx
{
    namespace bp {
        namespace GPUK {
            /**
            * @param proc_dims : ~.x -> width of the array to be filled, in vec4
            */
            __global__ void
                cu_fill2D_constant_v128_b32(float4* src, const float4 val, const uint2 proc_dims, const uint Wsrc);


            /**
            * @param proc_dims : ~.x -> width of the array to be filled, in vec4
            */
            __global__ void
                cu_fill2D_constant_v128_b32_LF(float4* src, const float4 val, const float4 _end_val, const uint2 proc_dims, const uint Wsrc);
        }

        void cu_fill2D_constant_v128_b32_caller(float* src, const float val, const uint2 proc_dims, const uint Wsrc, decx::cuda_stream* S);


        void cu_fill2D_constant_v128_b64_caller(double* src, const double val, const uint2 proc_dims, const uint Wsrc, decx::cuda_stream* S);
    }
}


#endif