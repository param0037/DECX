/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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