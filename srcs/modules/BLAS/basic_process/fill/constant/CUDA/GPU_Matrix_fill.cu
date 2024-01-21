/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "GPU_Matrix_fill.cuh"



_DECX_API_ de::DH de::cuda::Constant_fp32(de::GPU_Matrix& src, const float value)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::bp::cu_fill2D_constant_v128_b32_caller((float*)_src->Mat.ptr, value, make_uint2(_src->Width(), _src->Height()), _src->Pitch(), S);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    return handle;
}




_DECX_API_ de::DH de::cuda::Constant_int32(de::GPU_Matrix& src, const int value)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_INT32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::bp::cu_fill2D_constant_v128_b32_caller((float*)_src->Mat.ptr, *((float*)&value), make_uint2(_src->Width(), _src->Height()), _src->Pitch(), S);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    return handle;
}




_DECX_API_ de::DH de::cuda::Constant_fp64(de::GPU_Matrix& src, const double value)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_FP64_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::bp::cu_fill2D_constant_v128_b64_caller((double*)_src->Mat.ptr, value, make_uint2(_src->Width(), _src->Height()), _src->Pitch(), S);

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    return handle;
}