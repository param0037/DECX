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


#ifndef _REDUCE_CMP_CUH_
#define _REDUCE_CMP_CUH_


#include "reduce_kernel_utils.cuh"
#include "../../Scan_CUDA/scan.cuh"


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
            __global__ void cu_block_reduce_cmp1D_int32(const int4* __restrict src, int32_t* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1, const int32_t _fill_val);


            template <bool _is_max>
            __global__ void cu_block_reduce_cmp1D_fp64(const double2* __restrict src, double* __restrict dst,
                const uint64_t proc_len_v2, const uint64_t proc_len_v1, const double _fill_val);


            template <bool _is_max>
            __global__ void cu_block_reduce_cmp1D_u8(const int4* __restrict src, uint8_t* __restrict dst,
                const uint64_t proc_len_v4, const uint64_t proc_len_v1, const uint8_t _fill_val);


            template <bool _is_max>
            __global__ void cu_block_reduce_cmp1D_fp16(const float4* __restrict src, __half* __restrict dst,
                const uint64_t proc_len_v8, const uint64_t proc_len_v1, const __half _fill_val);
        }
    }
}



namespace decx
{
    namespace reduce
    {
        namespace GPUK
        {
            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_flatten_fp32(const float4* __restrict src, float* __restrict dst,
                const uint32_t Wsrc_v4, const uint2 proc_dims, const float _fill_val);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_flatten_int32(const int4* __restrict src, int32_t* __restrict dst,
                const uint32_t Wsrc_v4, const uint2 proc_dims, const int32_t _fill_val);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_flatten_fp64(const double2* __restrict src, double* __restrict dst,
                const uint32_t Wsrc_v4, const uint2 proc_dims, const double _fill_val);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_flatten_fp16(const float4* __restrict src, __half* __restrict dst,
                const uint32_t Wsrc_v8, const uint2 proc_dims, const __half _flil_val);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_flatten_u8(const int4* __restrict src, uint8_t* __restrict dst,
                const uint32_t Wsrc_v8, const uint2 proc_dims, const uint8_t _flil_val);

        }
    }
}



namespace decx
{
    namespace reduce
    {
        namespace GPUK 
        {
            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_h_fp32(const float4* __restrict src, float* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_h_fp16(const float4* __restrict src, __half* __restrict dst,
                const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_h_u8(const int4* __restrict src, uint8_t* __restrict dst,
                const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims);



            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_h_fp32_transp(const float4* __restrict src, float* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_h_fp16_transp(const float4* __restrict src, __half* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * The kernel calculates the reduced row sum (within a block, with 32 x 4 = 128 elements),
            * transposing the results, then store them in destinated buffer
            */
            void cu_warp_reduce_cmp2D_h_u8_transp(const int4* __restrict src, uint8_t* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims);



            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * process area[32 * 4, 8] = [128, 8]
            */
            void cu_warp_reduce_cmp2D_v_fp32(const float4* __restrict src, float4* __restrict dst,
                const uint32_t Wsrc_v4, uint32_t Wdst_v1, const uint2 proc_dims_v4);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * process area[32 * 4, 8] = [128, 8]
            */
            void cu_warp_reduce_cmp2D_v_fp16(const float4* __restrict src, float4* __restrict dst,
                const uint32_t Wsrc_v8, uint32_t Wdst_v1, const uint2 proc_dims_v4);


            template <bool _is_max> __global__
            /*
            * configure : thread[32, 8]
            * process area[32 * 4, 8] = [128, 8]
            */
            void cu_warp_reduce_cmp2D_v_u8(const int4* __restrict src, int4* __restrict dst,
                const uint32_t Wsrc_v16, uint32_t Wdst_v1, const uint2 proc_dims_v16);
        }
    }
}




#endif