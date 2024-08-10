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

#ifndef _GEMM_CPLXD_CALLERS_CUH_
#define _GEMM_CPLXD_CALLERS_CUH_


#include "../GEMM_kernels.cuh"


namespace decx
{
namespace blas{
    static void GEMM_cplxd_16_32_64(const void* A, const void* B, void* dst, const uint2 proc_dims_v1, 
        const uint32_t _L_v1, const uint32_t pitchA_v1, const uint32_t pitchB_v1, const uint32_t pitchdst_v1,
        decx::cuda_stream* S, const void* C, const de::CPd alpha, const de::CPd beta);
}
}


static void decx::blas::
GEMM_cplxd_16_32_64(const void* A,              const void* B, 
                   void* dst,                   const uint2 proc_dims_v1, 
                   const uint32_t _L_v1,        const uint32_t pitchA_v1, 
                   const uint32_t pitchB_v1,    const uint32_t pitchdst_v1,
                   decx::cuda_stream* S,        const void* C, 
                   const de::CPd alpha,         const de::CPd beta)
{
    dim3 thread(32, 8);
    dim3 grid(decx::utils::ceil<uint32_t>(proc_dims_v1.x, 32), 
              decx::utils::ceil<uint32_t>(proc_dims_v1.y, 64));

    if (C== NULL){
        decx::blas::GPUK::cu_GEMM_cplxd_kernel_16_32_64<<<grid, thread, 0, S->get_raw_stream_ref()>>>(
            (double2*)A, (double2*)B, (double2*)dst, proc_dims_v1, _L_v1, pitchA_v1, pitchB_v1, pitchdst_v1);
    }
    else{
        decx::blas::GPUK::cu_GEMM_cplxd_F_kernel_16_32_64<<<grid, thread, 0, S->get_raw_stream_ref()>>>(
            (double2*)A, (double2*)B, (double2*)C, (double2*)dst, alpha, beta, proc_dims_v1, _L_v1, pitchA_v1, 
            pitchB_v1, pitchdst_v1);
    }
}


#endif