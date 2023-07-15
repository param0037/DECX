/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "dot_fp16.h"


de::DH de::cuda::Dot_fp16(de::GPU_Vector& A, de::GPU_Vector& B, de::Half* res, const int accu_flag)
{
    decx::_GPU_Vector* _A = dynamic_cast<decx::_GPU_Vector*>(&A);
    decx::_GPU_Vector* _B = dynamic_cast<decx::_GPU_Vector*>(&B);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->length != _B->length) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        return handle;
    }

    decx::cuda::Dot_fp16_vec(_A, _B, res, &handle, accu_flag);
}


de::DH de::cuda::Dot_fp16(de::GPU_Matrix& A, de::GPU_Matrix& B, de::Half* res, const int accu_flag)
{
    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->Width() != _B->Width() || _A->Height() != _B->Height()) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        return handle;
    }

    decx::cuda::Dot_fp16_mat(_A, _B, res, &handle, accu_flag);
}



de::DH de::cuda::Dot_fp16(de::GPU_Tensor& A, de::GPU_Tensor& B, de::Half* res, const int accu_flag)
{
    decx::_GPU_Tensor* _A = dynamic_cast<decx::_GPU_Tensor*>(&A);
    decx::_GPU_Tensor* _B = dynamic_cast<decx::_GPU_Tensor*>(&B);

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->Width() != _B->Width() || _A->Height() != _B->Height() || _A->Depth() != _B->Depth()) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        return handle;
    }

    decx::cuda::Dot_fp16_ten(_A, _B, res, &handle, accu_flag);
}