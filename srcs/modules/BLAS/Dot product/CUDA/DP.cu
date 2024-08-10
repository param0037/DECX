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
#include "1D/DP1D_callers.cuh"
#include "Dot_Product.cuh"
#include "2D/DP2D_1way_callers.cuh"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../core/cudaStream_management/cudaStream_queue.h"


namespace decx
{
namespace blas
{
    static void _vector_dot_caller(decx::_Vector* A, decx::_Vector* B, de::Number* res, de::DH* handle, 
                                    const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
    
    static void _dev_vector_dot_caller(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::Number* res, 
                                    de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);


    static void matrix_dot2D_1way_fp32_selector(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst, de::DH* handle, 
                                    const de::REDUCE_METHOD _rd_method);

    static void matrix_dot2D_1way_fp16_selector(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst, de::DH* handle, 
                                    const de::REDUCE_METHOD _rd_method, const uint32_t _fp16_accu);
}
}



static void decx::blas::_vector_dot_caller(decx::_Vector* A, decx::_Vector* B, de::Number* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
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



static void decx::blas::_dev_vector_dot_caller(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::Number* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
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



_DECX_API_ void de::blas::cuda::Dot_product(de::Vector& A, de::Vector& B, de::Number& res, const uint32_t _fp16_accu)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);

    decx::blas::_vector_dot_caller(_A, _B, &res, de::GetLastError(), _fp16_accu);
}



_DECX_API_ void de::blas::cuda::Dot_product(de::GPU_Vector& A, de::GPU_Vector& B, de::Number& res, const uint32_t _fp16_accu)
{
    de::GetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_GPU_Vector* _A = dynamic_cast<decx::_GPU_Vector*>(&A);
    decx::_GPU_Vector* _B = dynamic_cast<decx::_GPU_Vector*>(&B);

    decx::blas::_dev_vector_dot_caller(_A, _B, &res, de::GetLastError(), _fp16_accu);
}


static void decx::blas::matrix_dot2D_1way_fp32_selector(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst, de::DH* handle, const de::REDUCE_METHOD _rd_method)
{
    if (_rd_method == de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::blas::matrix_dot_1way_fp32<true>(A, B, dst);
    }
    else if (_rd_method == de::REDUCE_METHOD::_REDUCE2D_V_) {
        decx::blas::matrix_dot_1way_fp32<false>(A, B, dst);
    }
    else {
        decx::err::handle_error_info_modify(handle,
            decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
    }
}



static void decx::blas::matrix_dot2D_1way_fp16_selector(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst, de::DH* handle, 
    const de::REDUCE_METHOD _rd_method, const uint32_t _fp16_accu)
{
    if (_rd_method == de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::blas::matrix_dot_1way_fp16<true>(A, B, dst, _fp16_accu);
    }
    else if (_rd_method == de::REDUCE_METHOD::_REDUCE2D_V_) {
        decx::blas::matrix_dot_1way_fp16<false>(A, B, dst, _fp16_accu);
    }
    else {
        decx::err::handle_error_info_modify(handle,
            decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
    }
}



_DECX_API_ void
de::blas::cuda::Dot_product(de::Matrix& A, de::Matrix& B, de::Vector& dst, const de::REDUCE_METHOD _rd_method, const uint32_t _fp16_accu)
{
    de::ResetLastError();

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::blas::matrix_dot2D_1way_fp32_selector(_A, _B, _dst, de::GetLastError(), _rd_method);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::blas::matrix_dot2D_1way_fp16_selector(_A, _B, _dst, de::GetLastError(), _rd_method, _fp16_accu);
        break;
    default:
        break;
    }
}