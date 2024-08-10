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
#include "../fp16/GEMM_fp16_callers.cuh"
#include "../../../../../../common/Basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "../../../../../../common/FP16/float_half_convert.h"


decx::ResourceHandle decx::blas::g_cu_GEMM_fp16_planner;


template<>
decx::blas::CGKP decx::blas::cuda_GEMM_LS_planner<de::Half>::_kernel_props[9] = 
{   // kernel ptr                                regs    shared  threads mio_compute     LWH                transposed_A
    {(void*)&decx::blas::GEMM_fp16_64_128_128,    126,    33792,  256,  1.5625e-2,  make_uint3(16, 128, 128),   0},
    {(void*)&decx::blas::GEMM_fp16_32_128_128,    128,    16640,  256,  1.5625e-2,  make_uint3(32, 128, 128),   0},
    // Kernels that require transposed form of matrix A
    {(void*)&decx::blas::GEMM_fp16_64_128_128_T,  125,    32768,  256,  1.5625e-2,  make_uint3(32, 128, 128),   1},
    // empty vessel
    {NULL, 0, 0, 0, 0, make_uint3(0, 0, 0), 0},
    {NULL, 0, 0, 0, 0, make_uint3(0, 0, 0), 0},
    {NULL, 0, 0, 0, 0, make_uint3(0, 0, 0), 0},
    {NULL, 0, 0, 0, 0, make_uint3(0, 0, 0), 0},
    {NULL, 0, 0, 0, 0, make_uint3(0, 0, 0), 0},
    {NULL, 0, 0, 0, 0, make_uint3(0, 0, 0), 0},
};


template<> void decx::blas::cuda_GEMM_LS_planner<de::Half>::
run(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, decx::cuda_stream* S)
{   
    const auto* _k_prop_ptr = &decx::blas::cuda_GEMM_LS_planner<de::Half>::_kernel_props[this->_kernel_id];

    auto* _kernel_ptr = (decx::blas::GPUK::_cu_GEMM_kernel_ptr<de::Half>*)_k_prop_ptr->_kernel_ptr;
    
    if (_k_prop_ptr->_transpose_A){
        decx::blas::transpose2D_b2((float4*)A->Mat.ptr, 
                                   (float4*)this->_AT._ptr.ptr, 
                                   make_uint2(this->_A_layout.height, this->_A_layout.width), 
                                   this->_A_layout.pitch, 
                                   this->_AT._dims.x, 
                                   S);

        (*_kernel_ptr)(this->_AT._ptr.ptr,  B->Mat.ptr, 
                      dst->Mat.ptr,         make_uint2(dst->Width(), dst->Height()), 
                      A->Width(),           this->_AT._dims.x, 
                      B->Pitch(),           dst->Pitch(), 
                      S,                    NULL, 
                      de::Float2Half(1),    de::Float2Half(1));
    }
    else{
        (*_kernel_ptr)(A->Mat.ptr,          B->Mat.ptr, 
                       dst->Mat.ptr,        make_uint2(dst->Width(), dst->Height()), 
                       A->Width(),          A->Pitch(), 
                       B->Pitch(),          dst->Pitch(), 
                       S,                   NULL, 
                       de::Float2Half(1),   de::Float2Half(1));
    }
}



template<> void 
decx::blas::cuda_GEMM_LS_planner<de::Half>::run(decx::_GPU_Matrix* A,     decx::_GPU_Matrix* B, 
                                          decx::_GPU_Matrix* C,     decx::_GPU_Matrix* dst,
                                          const de::Half alpha,     const de::Half beta, 
                                          decx::cuda_stream* S)
{
    const auto* _k_prop_ptr = &decx::blas::cuda_GEMM_LS_planner<de::Half>::_kernel_props[this->_kernel_id];

    auto* _kernel_ptr = (decx::blas::GPUK::_cu_GEMM_kernel_ptr<de::Half>*)_k_prop_ptr->_kernel_ptr;

    if (_k_prop_ptr->_transpose_A){
        decx::blas::transpose2D_b2((float4*)A->Mat.ptr, 
                                   (float4*)this->_AT._ptr.ptr, 
                                   make_uint2(this->_A_layout.height, this->_A_layout.width), 
                                   this->_A_layout.pitch, 
                                   this->_AT._dims.x, 
                                   S);

        (*_kernel_ptr)(this->_AT._ptr.ptr,  B->Mat.ptr, 
                      dst->Mat.ptr,         make_uint2(dst->Width(), dst->Height()), 
                      A->Width(),           this->_AT._dims.x, 
                      B->Pitch(),           dst->Pitch(), 
                      S,                    C->Mat.ptr, 
                      alpha,                beta);
    }
    else{
        (*_kernel_ptr)(A->Mat.ptr,     B->Mat.ptr, 
                       dst->Mat.ptr,   make_uint2(dst->Width(), dst->Height()), 
                       A->Width(),     A->Pitch(), 
                       B->Pitch(),     dst->Pitch(), 
                       S,              C->Mat.ptr, 
                       alpha,          beta);
    }
}
