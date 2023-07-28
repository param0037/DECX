/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _REDUCE_SUM_CUH_
#define _REDUCE_SUM_CUH_

#include "reduce_kernel_utils.cuh"
#include "../../scan/CUDA/scan.cuh"


namespace decx
{
    namespace reduce
    {
        namespace GPUK {
            /*
            * [32 * 8] (8 warps / block)
            */
            __global__ void cu_block_reduce_sum1D_fp32(const float4* __restrict src, float* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1);


            __global__ void cu_block_reduce_sum1D_int32(const int4* __restrict src, int* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1);


            __global__ void cu_block_reduce_sum1D_u8_i32(const int4* __restrict src, int* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1);


            __global__ void cu_block_reduce_sum1D_fp16_fp32(const float4* __restrict src, float* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1);
        }
    }
}


#endif