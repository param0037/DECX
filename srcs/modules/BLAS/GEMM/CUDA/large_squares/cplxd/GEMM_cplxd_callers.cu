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

#include "../common/cuda_GEMM_LS_planner.cuh"
#include "../cplxd/GEMM_cplxd_callers.cuh"
#include "../../../../../../common/Basic_process/transpose/CUDA/transpose_kernels.cuh"


decx::ResourceHandle decx::blas::g_cu_GEMM_cplxd_planner;


template<> void decx::blas::cuda_GEMM_LS_planner<de::CPd>::
run(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, decx::cuda_stream* S)
{
    decx::blas::GEMM_cplxd_16_32_64(A->Mat.ptr, B->Mat.ptr, dst->Mat.ptr, 
        make_uint2(B->Width(), A->Height()), A->Width(), A->Pitch(), B->Pitch(), dst->Pitch(), S,
        NULL, {1.0, 0}, {1.0, 0});
}


template<> void decx::blas::cuda_GEMM_LS_planner<de::CPd>::
run(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* C, decx::_GPU_Matrix* dst, 
        const de::CPd alpha, const de::CPd beta, decx::cuda_stream* S)
{
    decx::blas::GEMM_cplxd_16_32_64(A->Mat.ptr, B->Mat.ptr, dst->Mat.ptr, 
        make_uint2(B->Width(), A->Height()), A->Width(), A->Pitch(), B->Pitch(), dst->Pitch(), S,
        C, alpha, beta);
}