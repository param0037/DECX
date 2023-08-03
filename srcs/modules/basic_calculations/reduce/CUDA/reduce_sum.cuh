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



namespace decx
{
    namespace reduce
    {
        namespace GPUK 
        {
            __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_sum2D_h_fp32(const float4* __restrict src, float* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


            __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_sum2D_1D_fp32(const float4* __restrict src, float* __restrict dst,
                const uint32_t Wsrc_v4, const uint2 proc_dims);

            __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_sum2D_h_fp32_transp(const float4* __restrict src, float* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


            __global__
            /*
            * configure : thread[32, 8]
            * process area[32 * 4, 8] = [128, 8]
            */
            void cu_warp_reduce_sum2D_v_fp32(const float4* __restrict src, float4* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims_v4);
        }
    }
}


#endif