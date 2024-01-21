/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CUDA_GEMM_CPL16_CALLER_H_
#define _CUDA_GEMM_CPL16_CALLER_H_


#include "GEMM_fp16_kernel_callers.h"



namespace decx
{
    namespace cuda
    {
        template <bool _print>
        void GEMM_fp16_organizer(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle);


        template <bool _print>
        void GEMM_fp16_ABC_organizer(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH* handle);
    }
}



template <bool _print>
void decx::cuda::GEMM_fp16_organizer(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _dst, de::DH* handle)
{
    if (_A->Width() != _B->Height()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
        return;
    }

    _dst->re_construct(_A->Type(), _B->Width(), _A->Height());

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->Width() % 16) != 0) ? (decx::utils::ceil<uint>(_A->Width(), 16) * 16) : _A->Width();
    pitch_B = ((_B->Width() % 128) != 0) ? (decx::utils::ceil<uint>(_B->Width(), 128) * 128) : _B->Width();
    hA = ((_A->Height() % 128) != 0) ? (decx::utils::ceil<uint>(_A->Height(), 128) * 128) : _A->Height();

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<de::Half> DA, DB, Ddst;
    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(de::Half))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(de::Half))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(de::Half))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(de::Half), _A->Mat.ptr, _A->Pitch() * sizeof(de::Half),
        _A->Width() * sizeof(de::Half), _A->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(de::Half), _B->Mat.ptr, _B->Pitch() * sizeof(de::Half),
        _B->Width() * sizeof(de::Half), _B->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::hGEMM_part(DA.ptr, DB.ptr, Ddst.ptr, pitch_A, pitch_B, hA, S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst->Mat.ptr, _dst->Pitch() * sizeof(de::Half), Ddst.ptr, pitch_B * sizeof(de::Half),
        _B->Width() * sizeof(de::Half), _A->Height(), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    S->detach();
    E->detach();
}


template <bool _print>
void decx::cuda::GEMM_fp16_ABC_organizer(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _C, decx::_Matrix* _dst, de::DH* handle)
{
    if (_A->Width() != _B->Height()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
        return;
    }

    if (_A->Height() != _C->Height() || _B->Width() != _C->Width()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
        return;
    }

    _dst->re_construct(_A->Type(), _B->Width(), _A->Height());

    int pitch_A,    // x16
        pitch_B,    // x128
        hA;            // x128

    // let hB = pitch_A, which means hB and pitch_A are x16 aligned
    pitch_A = ((_A->Width() % 16) != 0) ? (decx::utils::ceil<uint>(_A->Width(), 16) * 16) : _A->Width();
    pitch_B = ((_B->Width() % 128) != 0) ? (decx::utils::ceil<uint>(_B->Width(), 128) * 128) : _B->Width();
    hA = ((_A->Height() % 128) != 0) ? (decx::utils::ceil<uint>(_A->Height(), 128) * 128) : _A->Height();

    const size_t mem_A = static_cast<size_t>(pitch_A) * static_cast<size_t>(hA);
    const size_t mem_B = static_cast<size_t>(pitch_B) * static_cast<size_t>(pitch_A);
    const size_t mem_dst = static_cast<size_t>(pitch_B) * static_cast<size_t>(hA);

    decx::PtrInfo<de::Half> DA, DB, Ddst;
    if (decx::alloc::_device_malloc(&DA, mem_A * sizeof(de::Half))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&DB, mem_B * sizeof(de::Half))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&Ddst, mem_dst * sizeof(de::Half))) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION,
            DEV_ALLOC_FAIL);
        return;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DA.ptr, pitch_A * sizeof(de::Half), _A->Mat.ptr, _A->Pitch() * sizeof(de::Half),
        _A->Width() * sizeof(de::Half), _A->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    // copy B from host to device
    checkCudaErrors(cudaMemcpy2DAsync(DB.ptr, pitch_B * sizeof(de::Half), _B->Mat.ptr, _B->Pitch() * sizeof(de::Half),
        _B->Width() * sizeof(de::Half), _B->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    // contemparay storage to dst, to save device memory
    checkCudaErrors(cudaMemcpy2DAsync(Ddst.ptr, pitch_B * sizeof(de::Half), _C->Mat.ptr, _C->Pitch() * sizeof(de::Half),
        _C->Width() * sizeof(de::Half), _C->Height(), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::hGEMM_part_ABC(DA.ptr, DB.ptr, Ddst.ptr, Ddst.ptr, pitch_A, pitch_B, hA, S);

    // copy A from host to device
    checkCudaErrors(cudaMemcpy2DAsync(_dst->Mat.ptr, _dst->Pitch() * sizeof(de::Half), Ddst.ptr, pitch_B * sizeof(de::Half),
        _B->Width() * sizeof(de::Half), _A->Height(), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    decx::alloc::_device_dealloc(&DA);
    decx::alloc::_device_dealloc(&DB);
    decx::alloc::_device_dealloc(&Ddst);

    S->detach();
    E->detach();
}




namespace decx
{
    namespace cuda
    {
        template <bool _print>
        void GEMM_on_GPU_fp16(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, const int flag, de::DH* handle);


        template <bool _print>
        void GEMM_on_GPU_fp16_ABC(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* C, decx::_GPU_Matrix* dst, const int flag, de::DH* handle);
    }
}



template <bool _print>
void decx::cuda::GEMM_on_GPU_fp16(decx::_GPU_Matrix* _A, decx::_GPU_Matrix* _B, decx::_GPU_Matrix* _dst, const int flag, de::DH* handle)
{
    if (_A->Width() != _B->Height()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
        return;
    }

    if (_A->Height() != _dst->Height() || _B->Width() != _dst->Width()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
        return;
    }

    _dst->re_construct(_A->Type(), _B->Width(), _A->Height());

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::dev_hGEMM_caller_overall(_A, _B, _dst, S, flag);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _print>
void decx::cuda::GEMM_on_GPU_fp16_ABC(decx::_GPU_Matrix* _A, decx::_GPU_Matrix* _B, decx::_GPU_Matrix* _C, decx::_GPU_Matrix* _dst, 
    const int flag, de::DH* handle)
{
    if (_A->Width() != _B->Height()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
        return;
    }

    _dst->re_construct(_A->Type(), _B->Width(), _A->Height());

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::dev_hGEMM_ABC_caller_overall(_A, _B, _C, _dst, S, flag);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}




#endif