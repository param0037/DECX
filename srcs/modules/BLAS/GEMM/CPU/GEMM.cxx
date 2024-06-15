/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "fp32/GEMM_fp32_caller.h"
#include "64b/GEMM_64b_caller.h"
#include "GEMM.h"


_DECX_API_
de::DH de::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success<true>(&handle);

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::GEMM_fp32(_A, _B, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_64b<false>(_A, _B, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cpu::GEMM_64b<true>(_A, _B, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}



_DECX_API_
de::DH de::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }


    decx::err::Success(&handle);

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::GEMM_fp32_ABC(_A, _B, _C, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_64b_ABC<false>(_A, _B, _C, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cpu::GEMM_64b_ABC<true>(_A, _B, _C, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}
