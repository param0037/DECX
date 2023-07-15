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
#include "fp16/GEMM_fp16_caller.h"
#include "cpl32/GEMM_cpl32_caller.h"
#include "../../../../classes/classes_util.h"
#include "GEMM.h"


_DECX_API_
de::DH de::cuda::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_fp32_organizer<true>(_A, _B, _dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_fp16_organizer<true>(_A, _B, _dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_cpl32_organizer<true>(_A, _B, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_
de::DH de::cuda::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_fp32_ABC_organizer<true>(_A, _B, _C, _dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_fp16_ABC_organizer<true>(_A, _B, _C, _dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_cpl32_ABC_organizer<true>(_A, _B, _C, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ void 
decx::cuda::GEMM_AB_Raw_API(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle)
{
    switch (A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_fp32_organizer<false>(A, B, dst, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_fp16_organizer<false>(A, B, dst, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_cpl32_organizer<false>(A, B, dst, handle);
        break;
    default:
        break;
    }
}



_DECX_API_ void 
decx::cuda::GEMM_ABC_Raw_API(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH* handle)
{
    switch (A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_fp32_ABC_organizer<false>(A, B, C, dst, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_fp16_ABC_organizer<false>(A, B, C, dst, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_cpl32_ABC_organizer<false>(A, B, C, dst, handle);
        break;
    default:
        break;
    }
}



// --------------------------------------------- pure GPU ------------------------------------------------


_DECX_API_
de::DH de::cuda::GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst, const int flag)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_on_GPU_fp32<true>(_A, _B, _dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_on_GPU_fp16<true>(_A, _B, _dst, flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_on_GPU_cpl32<true>(_A, _B, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_
de::DH de::cuda::GEMM(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst, const int flag)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);
    decx::_GPU_Matrix* _C = dynamic_cast<decx::_GPU_Matrix*>(&C);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_on_GPU_fp32_ABC<true>(_A, _B, _C, _dst, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_on_GPU_fp16_ABC<true>(_A, _B, _C, _dst, flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_on_GPU_cpl32_ABC<true>(_A, _B, _C, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ void 
decx::cuda::dev_GEMM_AB_Raw_API(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, const int flag, de::DH* handle)
{
    switch (A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_on_GPU_fp32<false>(A, B, dst, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_on_GPU_fp16<false>(A, B, dst, flag, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_on_GPU_cpl32<false>(A, B, dst, handle);
        break;
    default:
        break;
    }
}



_DECX_API_ void 
decx::cuda::dev_GEMM_ABC_Raw_API(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* C, decx::_GPU_Matrix* dst, const int flag, de::DH* handle)
{
    switch (A->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::GEMM_on_GPU_fp32_ABC<false>(A, B, C, dst, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::GEMM_on_GPU_fp16_ABC<false>(A, B, C, dst, flag, handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::cuda::GEMM_on_GPU_cpl32_ABC<false>(A, B, C, dst, handle);
        break;
    default:
        break;
    }
}