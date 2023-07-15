/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../Add_kernel.cuh"
#include "vector_operators.h"


de::DH de::cuda::Add(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst)
{
    decx::_GPU_Vector& _A = dynamic_cast<decx::_GPU_Vector&>(A);
    decx::_GPU_Vector& _B = dynamic_cast<decx::_GPU_Vector&>(B);
    decx::_GPU_Vector& _dst = dynamic_cast<decx::_GPU_Vector&>(dst);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }
    if (_A._length != _B._length) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    switch (_A.type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kadd_m((de::Half*)_A.Vec.ptr, (de::Half*)_B.Vec.ptr, (de::Half*)_dst.Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kadd_m((float*)_A.Vec.ptr, (float*)_B.Vec.ptr, (float*)_dst.Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kadd_m((int*)_A.Vec.ptr, (int*)_B.Vec.ptr, (int*)_dst.Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kadd_m((double*)_A.Vec.ptr, (double*)_B.Vec.ptr, (double*)_dst.Vec.ptr, len, S);
        break;
    default:
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(&handle);
    return handle;
}




de::DH de::cuda::Add(de::GPU_Vector& A, void* __x, de::GPU_Vector& dst)
{
    decx::_GPU_Vector& _A = dynamic_cast<decx::_GPU_Vector&>(A);
    decx::_GPU_Vector& _dst = dynamic_cast<decx::_GPU_Vector&>(dst);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(&handle);
        return handle;
    }

    if (__x == NULL) {
        decx::err::InvalidParam(&handle);
        return handle;
    }

    const size_t len = (size_t)_A._length;
    switch (_A.type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kadd_c((de::Half*)_A.Vec.ptr, *((de::Half*)__x), (de::Half*)_dst.Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kadd_c((float*)_A.Vec.ptr, *((float*)__x), (float*)_dst.Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kadd_c((int*)_A.Vec.ptr, *((int*)__x), (int*)_dst.Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kadd_c((double*)_A.Vec.ptr, *((double*)__x), (double*)_dst.Vec.ptr, len, S);
        break;
    default:
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ void 
decx::cuda::cuda_Add_Raw_API(decx::_GPU_Vector* A, decx::_GPU_Vector* B, decx::_GPU_Vector* dst, de::DH* handle)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<false>(handle);
        return;
    }

    if (A->_length != B->_length) {
        decx::err::Mat_Dim_Not_Matching<false>(handle);
        return;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<false>(handle);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<false>(handle);
        return;
    }

    const size_t len = (size_t)A->_length;
    switch (A->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kadd_m((de::Half*)A->Vec.ptr, (de::Half*)B->Vec.ptr, (de::Half*)dst->Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kadd_m((float*)A->Vec.ptr, (float*)B->Vec.ptr, (float*)dst->Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kadd_m((int*)A->Vec.ptr, (int*)B->Vec.ptr, (int*)dst->Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kadd_m((double*)A->Vec.ptr, (double*)B->Vec.ptr, (double*)dst->Vec.ptr, len, S);
        break;
    default:
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(handle);
}




_DECX_API_ void 
decx::cuda::cuda_AddC_Raw_API(decx::_GPU_Vector* src, void* __x, decx::_GPU_Vector* dst, de::DH* handle)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<false>(handle);
        return;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<false>(handle);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<false>(handle);
        return;
    }

    if (__x == NULL) {
        decx::err::InvalidParam(handle);
        return;
    }

    const size_t len = (size_t)src->_length;
    switch (src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kadd_c((de::Half*)src->Vec.ptr, *((de::Half*)__x), (de::Half*)dst->Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kadd_c((float*)src->Vec.ptr, *((float*)__x), (float*)dst->Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kadd_c((int*)src->Vec.ptr, *((int*)__x), (int*)dst->Vec.ptr, len, S);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kadd_c((double*)src->Vec.ptr, *((double*)__x), (double*)dst->Vec.ptr, len, S);
        break;
    default:
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::err::Success(handle);
}