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

#include "arithmetic.h"
#include "arithmetic_callers_LUT.h"


_DECX_API_ void de::blas::cuda::
Arithmetic(de::InputGPUMatrix A, de::InputGPUMatrix B, de::OutputGPUMatrix dst, const int32_t arith_flag)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    const decx::_GPU_Matrix* _A = dynamic_cast<const decx::_GPU_Matrix*>(&A);
    const decx::_GPU_Matrix* _B = dynamic_cast<const decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    if ((arith_flag > de::MAX && arith_flag < de::SUB) || 
        (arith_flag > de::MAX | de::OP_INV && arith_flag < de::SUB | de::OP_INV)){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Sine or Cosine are not binary operators");
    }
    else{
        decx::blas::mat_arithmetic_caller_VVO(_A, _B, _dst, arith_flag, S, de::GetLastError());
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



_DECX_API_ void de::blas::cuda::
Arithmetic(de::InputGPUMatrix src, de::OutputGPUMatrix dst, const int32_t arith_flag)
{
    de::ResetLastError();

    const decx::_GPU_Matrix* _src = dynamic_cast<const decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL){
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if ((arith_flag > de::MAX && arith_flag < de::SUB) || 
        (arith_flag > de::MAX | de::OP_INV && arith_flag < de::SUB | de::OP_INV)){
        decx::blas::mat_arithmetic_caller_VO(_src, _dst, arith_flag, S, de::GetLastError());
    }
    else{
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Sine or Cosine are not binary operators");
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}
