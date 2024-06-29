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


#include "Dot_Product.cuh"
#include "DP1D_callers.cuh"


namespace decx
{
    namespace blas
    {
        template <bool _async_call>
        static void _vector_dot_caller(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
        template <bool _async_call>
        static void _dev_vector_dot_caller(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
    }
}



template <bool _async_call>
static void decx::blas::_vector_dot_caller(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::blas::vector_dot_fp32(A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::blas::vector_dot_fp16(A, B, res, _fp16_accu);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        decx::blas::vector_dot_fp64(A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::blas::vector_dot_cplxf(A, B, res);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}



template <bool _async_call>
static void decx::blas::_dev_vector_dot_caller(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::blas::dev_vector_dot_fp32(A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::blas::dev_vector_dot_fp16(A, B, res, _fp16_accu);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        decx::blas::dev_vector_dot_fp64(A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::blas::dev_vector_dot_cplxf(A, B, res);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}



_DECX_API_ void de::blas::cuda::Dot_product(de::Vector& A, de::Vector& B, de::DecxNumber& res, const uint32_t _fp16_accu)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);

    decx::blas::_vector_dot_caller<false>(_A, _B, &res, de::GetLastError(), _fp16_accu);
}



_DECX_API_ void de::blas::cuda::Dot_product(de::GPU_Vector& A, de::GPU_Vector& B, de::DecxNumber& res, const uint32_t _fp16_accu)
{
    de::GetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_GPU_Vector* _A = dynamic_cast<decx::_GPU_Vector*>(&A);
    decx::_GPU_Vector* _B = dynamic_cast<decx::_GPU_Vector*>(&B);

    decx::blas::_dev_vector_dot_caller<false>(_A, _B, &res, de::GetLastError(), _fp16_accu);
}


#ifdef _SELECTED_CALL_P3_
#undef _SELECTED_CALL_P3_
#endif

#ifdef _SELECTED_CALL_P4_
#undef _SELECTED_CALL_P4_
#endif