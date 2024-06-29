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


#include "_mm128_fp32_int32.cuh"



__global__ void
decx::type_cast::GPUK::cu_mm128_cvtfp32_i321D(const float4* __restrict      src,
                                              int4* __restrict              dst, 
                                              const size_t                  proc_len)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    decx::utils::_cuda_vec128 recv, store;

    if (tid < proc_len) {
        recv._vf = src[tid];
        
        store._vi.x = __float2int_rn(recv._vf.x);
        store._vi.y = __float2int_rn(recv._vf.y);
        store._vi.z = __float2int_rn(recv._vf.z);
        store._vi.w = __float2int_rn(recv._vf.w);
        
        dst[tid] = store._vi;
    }
}




__global__ void
decx::type_cast::GPUK::cu_mm128_cvti32_fp321D(const int4* __restrict        src,
                                              float4* __restrict            dst, 
                                              const size_t                  proc_len)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    decx::utils::_cuda_vec128 recv, store;

    if (tid < proc_len) {
        recv._vi = src[tid];
        
        store._vf.x = __int2float_rn(recv._vi.x);
        store._vf.y = __int2float_rn(recv._vi.y);
        store._vf.z = __int2float_rn(recv._vi.z);
        store._vf.w = __int2float_rn(recv._vi.w);
        
        dst[tid] = store._vf;
    }
}





void 
decx::type_cast::_mm128_cvtfp32_i32_caller1D(const float4*           src, 
                                                int4*                dst, 
                                                const size_t            proc_len, 
                                                decx::cuda_stream*      S)
{
    const uint block_length = decx::cuda::_get_cuda_prop().maxThreadsPerBlock;
    decx::type_cast::GPUK::cu_mm128_cvtfp32_i321D
        << <decx::utils::ceil<size_t>(proc_len, block_length), block_length, 0, S->get_raw_stream_ref() >> > (src, dst, proc_len);
}


void 
decx::type_cast::_mm128_cvti32_fp32_caller1D(const int4*            src, 
                                             float4*                dst, 
                                             const size_t           proc_len, 
                                             decx::cuda_stream*     S)
{
    const uint block_length = decx::cuda::_get_cuda_prop().maxThreadsPerBlock;
    decx::type_cast::GPUK::cu_mm128_cvti32_fp321D
        << <decx::utils::ceil<size_t>(proc_len, block_length), block_length, 0, S->get_raw_stream_ref() >> > (src, dst, proc_len);
}

