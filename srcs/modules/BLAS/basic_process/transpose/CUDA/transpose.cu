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
        decx::blas::transpose2D_b4((float2*)_src->Mat.ptr, (float2*)_dst->Mat.ptr,
            make_uint2(_dst->Width(), _dst->Height()), _src->Pitch(), _dst->Pitch(), S);
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_ || _src->Type() == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::blas::transpose2D_b8((double2*)_src->Mat.ptr, (double2*)_dst->Mat.ptr,
            make_uint2(_dst->Width(), _dst->Height()), _src->Pitch(), _dst->Pitch(), S);
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        decx::blas::transpose2D_b1((uint32_t*)_src->Mat.ptr, (uint32_t*)_dst->Mat.ptr,
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
        decx::blas::transpose2D_b4_dense((float*)_src->Vec.ptr, (float*)_dst->Vec.ptr, make_uint2(1133, 1011), 1011, 1133, S);
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