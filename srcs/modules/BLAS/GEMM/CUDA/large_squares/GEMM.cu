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


#include "fp32/GEMM_fp32_caller.h"
#include "fp16/GEMM_fp16_caller.h"
#include "cpl32/GEMM_cpl32_caller.h"
#include "../../../../classes/classes_util.h"
#include "GEMM.h"


_DECX_API_ void 
de::blas::cuda::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_fp32_organizer<true>(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_fp16_organizer<true>(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_cpl32_organizer<true>(_A, _B, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}


_DECX_API_ void 
de::blas::cuda::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_fp32_ABC_organizer<true>(_A, _B, _C, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_fp16_ABC_organizer<true>(_A, _B, _C, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_cpl32_ABC_organizer<true>(_A, _B, _C, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}


// --------------------------------------------- pure GPU ------------------------------------------------


_DECX_API_ void 
de::blas::cuda::GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_on_GPU_fp32<true>(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_on_GPU_fp16<true>(_A, _B, _dst, flag, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_on_GPU_cpl32<true>(_A, _B, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}


_DECX_API_ void 
de::blas::cuda::GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, const int flag)
{
    de::GetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _C = dynamic_cast<decx::_GPU_Matrix*>(&C);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_on_GPU_fp32_ABC<true>(_A, _B, _C, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_on_GPU_fp16_ABC<true>(_A, _B, _C, _dst, flag, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_on_GPU_cpl32_ABC<true>(_A, _B, _C, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}
