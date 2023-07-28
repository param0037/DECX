/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _REDUCE_CMP_CUH_
#define _REDUCE_CMP_CUH_


#include "reduce_kernel_utils.cuh"
#include "../../scan/CUDA/scan.cuh"


namespace decx
{
    namespace reduce
    {
        namespace GPUK 
        {
            template <bool _is_max>
            __global__ void cu_block_reduce_cmp1D_fp32(const float4* __restrict src, float* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1, const float _fill_val);


            template <bool _is_max>
            __global__ void cu_block_reduce_cmp1D_u8(const float4* __restrict src, uint8_t* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1, const uint8_t _fill_val);


            template <bool _is_max>
            __global__ void cu_block_reduce_cmp1D_fp16(const float4* __restrict src, __half* __restrict dst,
                const uint64_t proc_len_v8, const uint64_t proc_len_v1, const __half _fill_val);
        }
    }
}


#endif