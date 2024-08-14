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


#include "matrix_B_arrange.h"
#include "GEMM_callers.h"


namespace de
{
namespace blas {
    namespace cpu {
        _DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);
    }
}
}



_DECX_API_ void de::blas::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst)
{
    de::ResetLastError();

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::blas::GEMM_fp32<false>(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::blas::GEMM_64b<false, false>(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        //decx::blas::GEMM_64b<false, true>(_A, _B, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        //decx::blas::GEMM_cplxd<false>(_A, _B, _dst, de::GetLastError());
        break;

    default:
        break;
    }
}



_DECX_API_ void de::blas::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix &C, de::Matrix& dst)
{
    de::ResetLastError();

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::blas::GEMM_fp32<true>(_A, _B, _dst, de::GetLastError(), _C);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::blas::GEMM_64b<true, false>(_A, _B, _dst, de::GetLastError(), _C);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        //decx::blas::GEMM_64b<true, true>(_A, _B, _dst, de::GetLastError(), _C);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        //decx::blas::GEMM_cplxd<true>(_A, _B, _dst, de::GetLastError(), _C);
        break;

    default:
        break;
    }
}
