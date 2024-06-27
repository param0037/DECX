/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
        decx::blas::GEMM_64b<false, true>(_A, _B, _dst, de::GetLastError());
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
        decx::blas::GEMM_64b<true, true>(_A, _B, _dst, de::GetLastError(), _C);
        break;

    default:
        break;
    }
}
