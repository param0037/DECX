/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _MM128_FP32_INT32_CUH_
#define _MM128_FP32_INT32_CUH_


#include "../../../../core/cudaStream_management/cudaStream_package.h"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../core/configs/config.h"


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
            cu_mm128_cvtfp32_i321D(const float4* src, int4* dst, const size_t proc_len);


            /**
            * @param src : pointer of the input matrix
            * @param dst : pointer of the output matrix
            * @param proc_len : length of processed area, in vec4
            */
            __global__ void
            cu_mm128_cvti32_fp321D(const int4* src, float4* dst, const size_t proc_len);
        }

        void _mm128_cvtfp32_i32_caller1D(const float4* src, int4* dst, const size_t proc_len, decx::cuda_stream* S);


        void _mm128_cvti32_fp32_caller1D(const int4* src, float4* dst, const size_t proc_len, decx::cuda_stream* S);
    }
}


#endif