/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../Mul_kernel.cuh"
#include "../../../core/basic.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_  de::DH Mul(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst);


        _DECX_API_  de::DH Mul(de::GPU_Vector& src, void* __x, de::GPU_Vector& dst);
    }
}


de::DH de::cuda::Mul(de::GPU_Vector& A, de::GPU_Vector& B, de::GPU_Vector& dst)
{
    decx::_GPU_Vector& _A = dynamic_cast<decx::_GPU_Vector&>(A);
    decx::_GPU_Vector& _B = dynamic_cast<decx::_GPU_Vector&>(B);
    decx::_GPU_Vector& _dst = dynamic_cast<decx::_GPU_Vector&>(dst);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    if (_A._length != _B._length) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
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

    const size_t len = (size_t)_A._length;
    switch (_A.type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kmul_m((de::Half*)_A.Vec.ptr, (de::Half*)_B.Vec.ptr, (de::Half*)_dst.Vec.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kmul_m((float*)_A.Vec.ptr, (float*)_B.Vec.ptr, (float*)_dst.Vec.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kmul_m((int*)_A.Vec.ptr, (int*)_B.Vec.ptr, (int*)_dst.Vec.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kmul_m((double*)_A.Vec.ptr, (double*)_B.Vec.ptr, (double*)_dst.Vec.ptr, len, S);
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





de::DH de::cuda::Mul(de::GPU_Vector& A, void* __x, de::GPU_Vector& dst)
{
    decx::_GPU_Vector& _A = dynamic_cast<decx::_GPU_Vector&>(A);
    decx::_GPU_Vector& _dst = dynamic_cast<decx::_GPU_Vector&>(dst);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    if (_A.length != _dst.length) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching, MAT_DIM_NOT_MATCH);
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

    const size_t len = (size_t)_A._length;
    switch (_A.type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kmul_c((de::Half*)_A.Vec.ptr, *((de::Half*)__x), (de::Half*)_dst.Vec.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kmul_c((float*)_A.Vec.ptr, *((float*)__x), (float*)_dst.Vec.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kmul_c((int*)_A.Vec.ptr, *((int*)__x), (int*)_dst.Vec.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kmul_c((double*)_A.Vec.ptr, *((double*)__x), (double*)_dst.Vec.ptr, len, S);
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
