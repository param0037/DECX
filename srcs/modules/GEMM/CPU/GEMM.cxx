/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "fp32/GEMM_fp32_caller.h"
#include "fp64/GEMM_fp64_caller.h"


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);
    }
}


_DECX_API_
de::DH de::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    switch (A.Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::GEMM_fp32(A, B, dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_fp64(A, B, dst, &handle);
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

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    switch (A.Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::GEMM_fp32_ABC(A, B, C, dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::GEMM_fp64_ABC(A, B, C, dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}