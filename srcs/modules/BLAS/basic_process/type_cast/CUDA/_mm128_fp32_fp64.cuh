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


#ifndef _MM128_FP32_FP64_CUH_
#define _MM128_FP32_FP64_CUH_


#include "../../../../core/cudaStream_management/cudaStream_package.h"


namespace decx
{
    namespace type_cast {
        namespace GPUK {
            /**
            * @param src : pointer of the input matrix
            * @param dst : pointer of the output matrix
            * @param proc_len : length of processed area, in vec4
            */
            __global__ void
            cu_mm128_cvtfp32_fp641D(const float4* src, double2* dst, const size_t proc_len);


            /**
            * @param src : pointer of the input matrix
            * @param dst : pointer of the output matrix
            * @param proc_len : length of processed area, in vec4
            */
            __global__ void
            cu_mm128_cvtfp64_fp321D(const double2* src, float4* dst, const size_t proc_len);
        }

        void _mm128_cvtfp32_fp64_caller1D(const float4* src, double2* dst, const size_t proc_len, decx::cuda_stream* S);
        
        
        void _mm128_cvtfp64_fp32_caller1D(const double2* src, float4* dst, const size_t proc_len, decx::cuda_stream* S);
    }



    namespace type_cast {
        namespace GPUK {
            /**
            * @param src : pointer of the input matrix
            * @param dst : pointer of the output matrix
            * @param proc_dims : ~.x -> width of processed area, in vec4; ~.y -> height of processed area
            * @param Wsrc : width of source matrix, in element
            * @param Wdst : eidth of destinated matrix, in element
            */
            __global__ void
            cu_mm128_cvtfp32_fp642D(const float4* src, double2* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst);


            /**
            * @param src : pointer of the input matrix
            * @param dst : pointer of the output matrix
            * @param proc_dims : ~.x -> width of processed area, in vec4; ~.y -> height of processed area
            * @param Wsrc : width of source matrix, in element
            * @param Wdst : eidth of destinated matrix, in element
            */
            __global__ void
            cu_mm128_cvtfp64_fp322D(const double2* src, float4* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst);
        }

        void _mm128_cvtfp32_fp64_caller2D(
            const float4* src, double2* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S);


        void _mm128_cvtfp64_fp32_caller2D(
            const double2* src, float4* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst, decx::cuda_stream* S);
    }
}


#endif