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


#include "_mm128_fp32_fp64.cuh"
#include "../../../CUSV/decx_cuda_vectypes_ops.cuh"
#include "../../../../modules/core/configs/config.h"


__global__ void
decx::type_cast::GPUK::cu_mm128_cvtfp32_fp641D(const float4* __restrict     src, 
                                               double2* __restrict          dst, 
                                               const size_t                 proc_len)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    decx::utils::_cuda_vec128 recv, store0, store1;

    if (tid < proc_len) {
        recv._vf = src[tid];
        
        store0._vd.x = recv._vf.x;
        store0._vd.y = recv._vf.y;
        store1._vd.x = recv._vf.z;
        store1._vd.y = recv._vf.w;
        
        dst[tid * 2] = store1._vd;
        dst[tid * 2 + 1] = store1._vd;
    }
}



__global__ void
decx::type_cast::GPUK::cu_mm128_cvtfp64_fp321D(const double2* __restrict    src, 
                                               float4* __restrict           dst, 
                                               const size_t                 proc_len)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    decx::utils::_cuda_vec128 recv0, recv1, store;

    if (tid < proc_len) {
        recv0._vd = src[tid * 2];
        recv1._vd = src[tid * 2 + 1];

        store._vf.x = __double2float_rn(recv0._vd.x);
        store._vf.y = __double2float_rn(recv0._vd.y);
        store._vf.z = __double2float_rn(recv1._vd.x);
        store._vf.w = __double2float_rn(recv1._vd.y);

        dst[tid] = store._vf;
    }
}



void 
decx::type_cast::_mm128_cvtfp32_fp64_caller1D(const float4*           src, 
                                                double2*                dst, 
                                                const size_t            proc_len, 
                                                decx::cuda_stream*      S)
{
    const uint block_length = decx::cuda::_get_cuda_prop().maxThreadsPerBlock;
    decx::type_cast::GPUK::cu_mm128_cvtfp32_fp641D
        << <decx::utils::ceil<size_t>(proc_len, block_length), block_length, 0, S->get_raw_stream_ref() >> > (src, dst, proc_len);
}


void 
decx::type_cast::_mm128_cvtfp64_fp32_caller1D(const double2*          src, 
                                                float4*                 dst, 
                                                const size_t            proc_len, 
                                                decx::cuda_stream*      S)
{
    const uint block_length = decx::cuda::_get_cuda_prop().maxThreadsPerBlock;
    decx::type_cast::GPUK::cu_mm128_cvtfp64_fp321D
        << <decx::utils::ceil<size_t>(proc_len, block_length), block_length, 0, S->get_raw_stream_ref() >> > (src, dst, proc_len);
}



// ------------------------------------------------ 2D ---------------------------------------------------

__global__ void
decx::type_cast::GPUK::cu_mm128_cvtfp32_fp642D(const float4* __restrict         src, 
                                               double2* __restrict              dst, 
                                               const ulong2                     proc_dims, 
                                               const uint                       Wsrc, 
                                               const uint                       Wdst)
{
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const size_t dex_src = tidx * Wsrc + tidy,
        dex_dst = tidx * Wdst + tidy * 2;

    decx::utils::_cuda_vec128 recv, store0, store1;

    if (tidx < proc_dims.y && tidy < proc_dims.x) {
        recv._vf = src[dex_src];

        store0._vd.x = recv._vf.x;
        store0._vd.y = recv._vf.y;
        store1._vd.x = recv._vf.z;
        store1._vd.y = recv._vf.w;

        dst[dex_dst] = store1._vd;
        dst[dex_dst + 1] = store1._vd;
    }
}




__global__ void
decx::type_cast::GPUK::cu_mm128_cvtfp64_fp322D(const double2* __restrict        src, 
                                               float4* __restrict               dst, 
                                               const ulong2                     proc_dims,
                                               const uint                       Wsrc, 
                                               const uint                       Wdst)
{
    const uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const size_t dex_src = tidx * Wsrc + tidy * 2,
        dex_dst = tidx * Wdst + tidy;

    decx::utils::_cuda_vec128 recv0, recv1, store;

    if (tidx < proc_dims.y && tidy < proc_dims.x) {
        recv0._vd = src[dex_src];
        recv1._vd = src[dex_src + 1];

        store._vf.x = __double2float_rn(recv0._vd.x);
        store._vf.y = __double2float_rn(recv0._vd.y);
        store._vf.z = __double2float_rn(recv1._vd.x);
        store._vf.w = __double2float_rn(recv1._vd.y);
        
        dst[dex_dst] = store._vf;
    }
}


void decx::type_cast::_mm128_cvtfp32_fp64_caller2D(const float4*         src, 
                                  double2*              dst, 
                                  const ulong2          proc_dims, 
                                  const uint            Wsrc, 
                                  const uint            Wdst, 
                                  decx::cuda_stream*    S)
{
    dim3 block(16, 16);
    dim3 grid(decx::utils::ceil<size_t>(proc_dims.y, 16), decx::utils::ceil<size_t>(proc_dims.x, 16));

    decx::type_cast::GPUK::cu_mm128_cvtfp32_fp642D << <grid, block, 0, S->get_raw_stream_ref() >> > (src, dst, proc_dims, Wsrc, Wdst);
}



void decx::type_cast::_mm128_cvtfp64_fp32_caller2D(const double2*        src, 
                                                   float4*               dst, 
                                                   const ulong2          proc_dims, 
                                                   const uint            Wsrc, 
                                                   const uint            Wdst, 
                                                   decx::cuda_stream*    S)
{
    dim3 block(16, 16);
    dim3 grid(decx::utils::ceil<size_t>(proc_dims.y, 16), decx::utils::ceil<size_t>(proc_dims.x, 16));

    decx::type_cast::GPUK::cu_mm128_cvtfp64_fp322D << <grid, block, 0, S->get_raw_stream_ref() >> > (src, dst, proc_dims, Wsrc, Wdst);
}