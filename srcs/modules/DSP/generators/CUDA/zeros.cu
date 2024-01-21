/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "zeros.cuh"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../core/cudaStream_management/cudaStream_queue.h"


_DECX_API_ de::DH de::gen::cuda::Zeros(de::GPU_Vector& src)
{
    de::DH handle;
    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT, CLASS_NOT_INIT);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return handle;
    }

    decx::alloc::Memset_D(_src->Vec.block, _src->total_bytes, 0, S->get_raw_stream_ptr());

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH de::gen::cuda::Zeros(de::GPU_Matrix& src)
{
    de::DH handle;
    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT, CLASS_NOT_INIT);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return handle;
    }

    decx::alloc::Memset_D(_src->Mat.block, _src->get_total_bytes(), 0, S->get_raw_stream_ptr());

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::gen::cuda::Zeros(de::GPU_Tensor& src)
{
    de::DH handle;
    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT, CLASS_NOT_INIT);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::cuda_event* E = NULL; 
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return handle;
    }

    decx::alloc::Memset_D(_src->Tens.block, _src->total_bytes, 0, S->get_raw_stream_ptr());

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(&handle);
    return handle;
}