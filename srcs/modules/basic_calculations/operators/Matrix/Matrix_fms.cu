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


#include "../Fms_kernel.cuh"
#include "../../../core/basic.h"
#include "Matrix_operators.h"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../core/cudaStream_management/cudaStream_queue.h"



de::DH de::cuda::Fms(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst)
{
    decx::_GPU_Matrix& _A = dynamic_cast<decx::_GPU_Matrix&>(A);
    decx::_GPU_Matrix& _B = dynamic_cast<decx::_GPU_Matrix&>(B);
    decx::_GPU_Matrix& _C = dynamic_cast<decx::_GPU_Matrix&>(C);
    decx::_GPU_Matrix& _dst = dynamic_cast<decx::_GPU_Matrix&>(dst);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return handle;
    }

    if (_A.Width() != _B.Width() || _A.Height() != _B.Height()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return handle;
    }

    const uint64_t len = (uint64_t)_A.Pitch() * (uint64_t)_A.Height();
    switch (_A.Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kfms_m((de::Half*)_A.Mat.ptr, (de::Half*)_B.Mat.ptr, (de::Half*)_C.Mat.ptr, (de::Half*)_dst.Mat.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kfms_m((float*)_A.Mat.ptr, (float*)_B.Mat.ptr, (float*)_C.Mat.ptr, (float*)_dst.Mat.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kfms_m((int*)_A.Mat.ptr, (int*)_B.Mat.ptr, (int*)_C.Mat.ptr, (int*)_dst.Mat.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kfms_m((double*)_A.Mat.ptr, (double*)_B.Mat.ptr, (double*)_C.Mat.ptr, (double*)_dst.Mat.ptr, len, S);
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



de::DH de::cuda::Fms(de::GPU_Matrix& A, void* __x, de::GPU_Matrix& B, de::GPU_Matrix& dst)
{
    decx::_GPU_Matrix& _A = dynamic_cast<decx::_GPU_Matrix&>(A);
    decx::_GPU_Matrix& _B = dynamic_cast<decx::_GPU_Matrix&>(B);
    decx::_GPU_Matrix& _dst = dynamic_cast<decx::_GPU_Matrix&>(dst);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return handle;
    }

    if (_A.Width() != _B.Width() || _A.Height() != _B.Height()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::cuda_stream* S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return handle;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return handle;
    }

    const uint64_t len = (uint64_t)_A.Pitch() * (uint64_t)_A.Height();
    switch (_A.Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::dev_Kfms_c((de::Half*)_A.Mat.ptr, *(de::Half*)__x, (de::Half*)_B.Mat.ptr, (de::Half*)_dst.Mat.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::dev_Kfms_c((float*)_A.Mat.ptr, *(float*)__x, (float*)_B.Mat.ptr, (float*)_dst.Mat.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        decx::calc::dev_Kfms_c((int*)_A.Mat.ptr, *(int*)__x, (int*)_B.Mat.ptr, (int*)_dst.Mat.ptr, len, S);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::calc::dev_Kfms_c((double*)_A.Mat.ptr, *(double*)__x, (double*)_B.Mat.ptr, (double*)_dst.Mat.ptr, len, S);
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
