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
void decx::cpu::GEMM_AB_Raw_API(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _dst, de::DH* handle)
{
    decx::err::Success<false>(handle);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::GEMM_fp32<false>(_A, _B, _dst, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_64b<false, false>(_A, _B, _dst, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cpu::GEMM_64b<true, false>(_A, _B, _dst, handle);
        break;
    default:
        break;
    }
}


_DECX_API_
void decx::cpu::GEMM_ABC_Raw_API(decx::_Matrix* _A, decx::_Matrix* _B, decx::_Matrix* _C, decx::_Matrix* _dst, de::DH* handle)
{
    decx::err::Success<false>(handle);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::GEMM_fp32_ABC<false>(_A, _B, _C, _dst, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_64b_ABC<false, false>(_A, _B, _C, _dst, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cpu::GEMM_64b_ABC<true, false>(_A, _B, _C, _dst, handle);
        break;
    default:
        break;
    }
}


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
        decx::cpu::GEMM_fp32<true>(_A, _B, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_64b<false, true>(_A, _B, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cpu::GEMM_64b<true, true>(_A, _B, _dst, &handle);
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
        decx::cpu::GEMM_fp32_ABC<true>(_A, _B, _C, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_64b_ABC<false, true>(_A, _B, _C, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cpu::GEMM_64b_ABC<true, true>(_A, _B, _C, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}




_DECX_API_ void
de::cpu::GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& dst, de::DecxStream& S)
{
    de::DH handle;

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::async::register_async_task(S.Get_ID(), decx::cpu::GEMM_AB_Raw_API, _A, _B, _dst,
        S.Get_last_handle());
}


_DECX_API_ void
de::cpu::GEMM_Async(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst, de::DecxStream& S)
{
    de::DH handle;

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::async::register_async_task(S.Get_ID(), decx::cpu::GEMM_ABC_Raw_API, _A, _B, _C, _dst,
        S.Get_last_handle());
}