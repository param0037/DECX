/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CUDA_GEMM_FP32_CALLER_H_
#define _CUDA_GEMM_FP32_CALLER_H_


#include "GEMM_fp32_kernel_callers.h"


namespace decx
{
    namespace cuda
    {
        void GEMM_fp32_organizer(de::Matrix& A, de::Matrix& B, de::Matrix& dst, de::DH* handle);


        void GEMM_fp32_ABC_organizer(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst, de::DH* handle);
    }
}


void decx::cuda::GEMM_fp32_organizer(de::Matrix& A, de::Matrix& B, de::Matrix& dst, de::DH* handle)
{
    decx::_Matrix& _A = dynamic_cast<decx::_Matrix&>(A);
    decx::_Matrix& _B = dynamic_cast<decx::_Matrix&>(B);
    decx::_Matrix& _dst = dynamic_cast<decx::_Matrix&>(dst);

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(handle);
    }

    _dst.re_construct(_A.type, _B.width, _A.height, decx::DATA_STORE_TYPE::Page_Locked);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0) ? (_cu_ceil(_A.width, 16) * 16) : _A.width;
    pitch_B = ((_B.width % 128) != 0) ? (_cu_ceil(_B.width, 128) * 128) : _B.width;
    hA = ((_A.height % 128) != 0) ? (_cu_ceil(_A.height, 128) * 128) : _A.height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<float> DA, DB, Ddst;

    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }

    decx::cuda_stream* S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(float), _A.Mat.ptr, _A.pitch * sizeof(float),
        _A.width * sizeof(float), _A.height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(float), _B.Mat.ptr, _B.pitch * sizeof(float),
        _B.width * sizeof(float), _B.height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::sGEMM_part(DA.ptr, DB.ptr, Ddst.ptr, pitch_A, pitch_B, hA, S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(float), Ddst.ptr, pitch_B * sizeof(float),
        _B.width * sizeof(float), _A.height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());

    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    S->detach();
}



void decx::cuda::GEMM_fp32_ABC_organizer(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst, de::DH *handle)
{
    decx::_Matrix& _A = dynamic_cast<decx::_Matrix&>(A);
    decx::_Matrix& _B = dynamic_cast<decx::_Matrix&>(B);
    decx::_Matrix& _C = dynamic_cast<decx::_Matrix&>(C);
    decx::_Matrix& _dst = dynamic_cast<decx::_Matrix&>(dst);

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(handle);
        return;
    }

    if (_A.height != _C.height || _B.width != _C.width) {
        decx::GEMM_DimNMatch(handle);
        return;
    }

    _dst.re_construct(_A.type, _B.width, _A.height, decx::DATA_STORE_TYPE::Page_Locked);

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A.width % 16) != 0)    ? (_cu_ceil(_A.width, 16) * 16)        : _A.width;
    pitch_B = ((_B.width % 128) != 0)    ? (_cu_ceil(_B.width, 128) * 128)    : _B.width;
    hA        = ((_A.height % 128) != 0)    ? (_cu_ceil(_A.height, 128) * 128)    : _A.height;

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<float> DA, DB, Ddst;
    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(float))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(handle);
        return;
    }
    
    decx::cuda_stream* S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(float), _A.Mat.ptr, _A.pitch * sizeof(float),
        _A.width * sizeof(float), _A.height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(float), _B.Mat.ptr, _B.pitch * sizeof(float),
        _B.width * sizeof(float), _B.height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    // contemparay storage to dst, to save device memory
    checkCudaErrors(cudaMemcpy2DAsync(Ddst.ptr, pitch_B * sizeof(float), _C.Mat.ptr, _C.pitch * sizeof(float),
        _C.width * sizeof(float), _C.height, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::sGEMM_part_ABC(DA.ptr, DB.ptr, Ddst.ptr, Ddst.ptr, pitch_A, pitch_B, hA, S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst.Mat.ptr, _dst.pitch * sizeof(float), Ddst.ptr, pitch_B * sizeof(float),
        _B.width * sizeof(float), _A.height, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    checkCudaErrors(cudaDeviceSynchronize());
    
    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    S->detach();
}




namespace decx
{
    namespace cuda
    {
        void GEMM_on_GPU_fp32(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, de::DH* handle);



        void GEMM_on_GPU_fp32_ABC(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, de::DH* handle);
    }
}


void decx::cuda::GEMM_on_GPU_fp32(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, de::DH* handle)
{
    decx::_GPU_Matrix& _A = dynamic_cast<decx::_GPU_Matrix&>(A);
    decx::_GPU_Matrix& _B = dynamic_cast<decx::_GPU_Matrix&>(B);
    decx::_GPU_Matrix& _dst = dynamic_cast<decx::_GPU_Matrix&>(dst);

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(handle);
    }
    _dst.re_construct(_A.type, _B.width, _A.height);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, "CUDA stream acess failure\n");
    }

    decx::dev_sGEMM_part(&_A, &_B, &_dst, S);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();
}



void decx::cuda::GEMM_on_GPU_fp32_ABC(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, de::DH* handle)
{
    decx::_GPU_Matrix& _A = dynamic_cast<decx::_GPU_Matrix&>(A);
    decx::_GPU_Matrix& _B = dynamic_cast<decx::_GPU_Matrix&>(B);
    decx::_GPU_Matrix& _C = dynamic_cast<decx::_GPU_Matrix&>(C);
    decx::_GPU_Matrix& _dst = dynamic_cast<decx::_GPU_Matrix&>(dst);

    if (_A.width != _B.height) {
        decx::MDim_Not_Matching(handle);
    }

    _dst.re_construct(_A.type, _B.width, _A.height);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);

    decx::dev_sGEMM_part_ABC(&_A, &_B, &_C, &_dst, S);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();
}




#endif
