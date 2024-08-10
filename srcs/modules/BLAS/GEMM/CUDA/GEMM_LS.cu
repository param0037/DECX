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

#include "large_squares/GEMM_kernels.cuh"
#include "large_squares/common/cuda_GEMM_LS_planner.cuh"
#include "GEMM.h"
#include "../../../../common/FP16/float_half_convert.h"


namespace decx
{
namespace blas{
    static void GEMM_fp32(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, de::DH* handle,
        decx::_GPU_Matrix* C = NULL, const float alpha = 1, const float beta = 1);


    static void GEMM_fp16(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, de::DH* handle,
        decx::_GPU_Matrix* C = NULL, const de::Half alpha = de::Float2Half(1), const de::Half beta = de::Float2Half(1));


    static void GEMM_fp64(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, de::DH* handle,
        decx::_GPU_Matrix* C = NULL, const double alpha = 1.f, const double beta = 1.f);


    static void GEMM_cplxf(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, de::DH* handle,
        decx::_GPU_Matrix* C = NULL, const de::CPf alpha = de::CPf(1.f, 0), const de::CPf beta = de::CPf(1.f, 0));


    static void GEMM_cplxd(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, de::DH* handle,
        decx::_GPU_Matrix* C = NULL, const de::CPd alpha = de::CPd(1.0, 0), const de::CPd beta = de::CPd(1.0, 0));
}
}


static void decx::blas::GEMM_fp32(decx::_GPU_Matrix* A,     decx::_GPU_Matrix* B, 
                                  decx::_GPU_Matrix* dst,   de::DH* handle,
                                  decx::_GPU_Matrix* C, 
                                  const float alpha,        const float beta)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    dst->re_construct(A->Type(), B->Width(), A->Height(), S);

    if (decx::blas::g_cu_GEMM_fp32_planner._res_ptr == NULL){
        g_cu_GEMM_fp32_planner.RegisterResource(new decx::blas::cuda_GEMM_LS_planner<float>, 5,
            decx::blas::cuda_GEMM_LS_planner<float>::release);
    }

    decx::blas::g_cu_GEMM_fp32_planner.lock();

    auto* _planner = decx::blas::g_cu_GEMM_fp32_planner.get_resource_raw_ptr<decx::blas::cuda_GEMM_LS_planner<float>>();

    if (_planner->changed(&A->get_layout(), &B->get_layout())){
        _planner->plan(&A->get_layout(), &B->get_layout(), &dst->get_layout(), de::GetLastError(), S);
    }

    if (C == NULL) {
        _planner->run(A, B, dst, S);
    }
    else{
        _planner->run(A, B, C, dst, alpha, beta, S);
    }

    E->event_record(S);
    E->synchronize();

    decx::blas::g_cu_GEMM_fp32_planner.unlock();
    
    S->detach();
    E->detach();
}



static void decx::blas::GEMM_fp16(decx::_GPU_Matrix* A,     decx::_GPU_Matrix* B, 
                                  decx::_GPU_Matrix* dst,   de::DH* handle,
                                  decx::_GPU_Matrix* C, 
                                  const de::Half alpha,     const de::Half beta)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    dst->re_construct(A->Type(), B->Width(), A->Height(), S);

    if (decx::blas::g_cu_GEMM_fp16_planner._res_ptr == NULL){
        g_cu_GEMM_fp16_planner.RegisterResource(new decx::blas::cuda_GEMM_LS_planner<de::Half>, 5,
            decx::blas::cuda_GEMM_LS_planner<de::Half>::release);
    }

    decx::blas::g_cu_GEMM_fp16_planner.lock();

    auto* _planner = decx::blas::g_cu_GEMM_fp16_planner.get_resource_raw_ptr<decx::blas::cuda_GEMM_LS_planner<de::Half>>();

    if (_planner->changed(&A->get_layout(), &B->get_layout())){
        _planner->plan(&A->get_layout(), &B->get_layout(), &dst->get_layout(), de::GetLastError(), S);
    }

    if (C == NULL) {
        _planner->run(A, B, dst, S);
    }
    else{
        _planner->run(A, B, C, dst, alpha, beta, S);
    }

    E->event_record(S);
    E->synchronize();

    decx::blas::g_cu_GEMM_fp16_planner.unlock();
    
    S->detach();
    E->detach();
}



static void decx::blas::GEMM_fp64(decx::_GPU_Matrix* A,     decx::_GPU_Matrix* B, 
                                  decx::_GPU_Matrix* dst,   de::DH* handle,
                                  decx::_GPU_Matrix* C, 
                                  const double alpha,       const double beta)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    
    dst->re_construct(A->Type(), B->Width(), A->Height(), S);

    if (decx::blas::g_cu_GEMM_fp64_planner._res_ptr == NULL){
        g_cu_GEMM_fp64_planner.RegisterResource(new decx::blas::cuda_GEMM_LS_planner<double>, 5,
            decx::blas::cuda_GEMM_LS_planner<double>::release);
    }
    
    decx::blas::g_cu_GEMM_fp64_planner.lock();

    auto* _planner = decx::blas::g_cu_GEMM_fp64_planner.get_resource_raw_ptr<decx::blas::cuda_GEMM_LS_planner<double>>();
    // if (_planner->changed(&A->get_layout(), &B->get_layout())){
    //     _planner->plan(&A->get_layout(), &B->get_layout(), &dst->get_layout(), de::GetLastError(), S);
    // }

    if (C == NULL) {
        _planner->run(A, B, dst, S);
    }
    else{
        _planner->run(A, B, C, dst, alpha, beta, S);
    }

    E->event_record(S);
    E->synchronize();

    decx::blas::g_cu_GEMM_fp64_planner.unlock();
    
    S->detach();
    E->detach();
}



static void decx::blas::GEMM_cplxf(decx::_GPU_Matrix* A,     decx::_GPU_Matrix* B, 
                                  decx::_GPU_Matrix* dst,   de::DH* handle,
                                  decx::_GPU_Matrix* C, 
                                  const de::CPf alpha,       const de::CPf beta)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    dst->re_construct(A->Type(), B->Width(), A->Height(), S);

    if (decx::blas::g_cu_GEMM_fp64_planner._res_ptr == NULL){
        g_cu_GEMM_fp64_planner.RegisterResource(new decx::blas::cuda_GEMM_LS_planner<de::CPf>, 5,
            decx::blas::cuda_GEMM_LS_planner<de::CPf>::release);
    }

    decx::blas::g_cu_GEMM_fp64_planner.lock();

    auto* _planner = decx::blas::g_cu_GEMM_fp64_planner.get_resource_raw_ptr<decx::blas::cuda_GEMM_LS_planner<de::CPf>>();

    // if (_planner->changed(&A->get_layout(), &B->get_layout())){
    //     _planner->plan(&A->get_layout(), &B->get_layout(), &dst->get_layout(), de::GetLastError(), S);
    // }

    if (C == NULL) {
        _planner->run(A, B, dst, S);
    }
    else{
        _planner->run(A, B, C, dst, alpha, beta, S);
    }

    E->event_record(S);
    E->synchronize();

    decx::blas::g_cu_GEMM_fp64_planner.unlock();
    
    S->detach();
    E->detach();
}



static void decx::blas::GEMM_cplxd(decx::_GPU_Matrix* A,     decx::_GPU_Matrix* B, 
                                  decx::_GPU_Matrix* dst,   de::DH* handle,
                                  decx::_GPU_Matrix* C, 
                                  const de::CPd alpha,       const de::CPd beta)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    dst->re_construct(A->Type(), B->Width(), A->Height(), S);

    if (decx::blas::g_cu_GEMM_cplxd_planner._res_ptr == NULL){
        g_cu_GEMM_cplxd_planner.RegisterResource(new decx::blas::cuda_GEMM_LS_planner<de::CPd>, 5,
            decx::blas::cuda_GEMM_LS_planner<de::CPd>::release);
    }

    decx::blas::g_cu_GEMM_cplxd_planner.lock();

    auto* _planner = decx::blas::g_cu_GEMM_cplxd_planner.get_resource_raw_ptr<decx::blas::cuda_GEMM_LS_planner<de::CPd>>();

    // if (_planner->changed(&A->get_layout(), &B->get_layout())){
    //     _planner->plan(&A->get_layout(), &B->get_layout(), &dst->get_layout(), de::GetLastError(), S);
    // }

    if (C == NULL) {
        _planner->run(A, B, dst, S);
    }
    else{
        _planner->run(A, B, C, dst, alpha, beta, S);
    }

    E->event_record(S);
    E->synchronize();

    decx::blas::g_cu_GEMM_cplxd_planner.unlock();
    
    S->detach();
    E->detach();
}


_DECX_API_ void de::blas::cuda::GEMM(de::GPU_Matrix& A, 
                                     de::GPU_Matrix& B, 
                                     de::GPU_Matrix& dst)
{
    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::blas::cuda_GEMM_LS_planner<_CUDA_GEMM_LS_PLANNER_GENERAL_TYPE_>::validate(_A, _B, NULL, de::GetLastError());

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::blas::GEMM_fp32(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::blas::GEMM_fp16(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::blas::GEMM_fp64(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::blas::GEMM_cplxf(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::blas::GEMM_cplxd(_A, _B, _dst, de::GetLastError());
        break;
    
    default:
        break;
    }   
}


_DECX_API_ void 
de::blas::cuda::GEMM(de::GPU_Matrix& A, 
                     de::GPU_Matrix& B, 
                     de::GPU_Matrix& C, 
                     de::GPU_Matrix& dst,
                     const de::Number alpha, 
                     const de::Number beta)
{
    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _C = dynamic_cast<decx::_GPU_Matrix*>(&C);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);
    
    decx::blas::cuda_GEMM_LS_planner<_CUDA_GEMM_LS_PLANNER_GENERAL_TYPE_>::validate(_A, _B, _C, de::GetLastError(), &alpha, &beta);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::blas::GEMM_fp32(_A, _B, _dst, de::GetLastError(), _C, alpha.get_data_ref<float>(), beta.get_data_ref<float>());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::blas::GEMM_fp16(_A, _B, _dst, de::GetLastError(), _C, alpha.get_data_ref<de::Half>(), beta.get_data_ref<de::Half>());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::blas::GEMM_fp64(_A, _B, _dst, de::GetLastError(), _C, alpha.get_data_ref<double>(), beta.get_data_ref<double>());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::blas::GEMM_cplxf(_A, _B, _dst, de::GetLastError(), _C, alpha.get_data_ref<de::CPf>(), beta.get_data_ref<de::CPf>());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::blas::GEMM_cplxd(_A, _B, _dst, de::GetLastError(), _C, alpha.get_data_ref<de::CPd>(), beta.get_data_ref<de::CPd>());
        break;
    
    default:
        break;
    }
}