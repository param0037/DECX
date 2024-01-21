/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "transpose_kernels.cuh"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_Vector.h"



namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst);


        _DECX_API_ de::DH Transpose(de::GPU_Vector& src, de::GPU_Vector& dst);
    }
}


_DECX_API_ de::DH
de::cuda::Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);

    if (_src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_ || _src->Type() == de::_DATA_TYPES_FLAGS_::_INT32_) {
        decx::bp::transpose2D_b4((float2*)_src->Mat.ptr, (float2*)_dst->Mat.ptr,
            make_uint2(_dst->Width(), _dst->Height()), _src->Pitch(), _dst->Pitch(), S);
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_ || _src->Type() == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::bp::transpose2D_b8((double2*)_src->Mat.ptr, (double2*)_dst->Mat.ptr,
            make_uint2(_dst->Width(), _dst->Height()), _src->Pitch(), _dst->Pitch(), S);
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        decx::bp::transpose2D_b1((uint32_t*)_src->Mat.ptr, (uint32_t*)_dst->Mat.ptr,
            make_uint2(_dst->Width(), _dst->Height()), _src->Pitch(), _dst->Pitch(), S);
    }
    else {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM, INVALID_PARAM);
        S->detach();
        return handle;
    }

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    return handle;
}


// This is only for test

_DECX_API_ de::DH
de::cuda::Transpose(de::GPU_Vector& src, de::GPU_Vector& dst)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);

    if (_src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_ || _src->Type() == de::_DATA_TYPES_FLAGS_::_INT32_) {
        decx::bp::transpose2D_b4_dense((float*)_src->Vec.ptr, (float*)_dst->Vec.ptr, make_uint2(1133, 1011), 1011, 1133, S);
    }
    else {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM, INVALID_PARAM);
        S->detach();
        return handle;
    }

    checkCudaErrors(cudaDeviceSynchronize());

    S->detach();

    return handle;
}