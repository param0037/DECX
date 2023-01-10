/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "dot_fp16.h"


de::DH de::cuda::Dot_fp16(de::GPU_Vector& A, de::GPU_Vector& B, de::Half* res, const int accu_flag)
{
    decx::_GPU_Vector* _A = dynamic_cast<decx::_GPU_Vector*>(&A);
    decx::_GPU_Vector* _B = dynamic_cast<decx::_GPU_Vector*>(&B);

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->length != _B->length) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::cuda::Dot_fp16_vec(_A, _B, res, &handle, accu_flag);
}


de::DH de::cuda::Dot_fp16(de::GPU_Matrix& A, de::GPU_Matrix& B, de::Half* res, const int accu_flag)
{
    decx::_GPU_Matrix* _A = dynamic_cast<decx::_GPU_Matrix*>(&A);
    decx::_GPU_Matrix* _B = dynamic_cast<decx::_GPU_Matrix*>(&B);

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->width != _B->width || _A->height != _B->height) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::cuda::Dot_fp16_mat(_A, _B, res, &handle, accu_flag);
}



de::DH de::cuda::Dot_fp16(de::GPU_Tensor& A, de::GPU_Tensor& B, de::Half* res, const int accu_flag)
{
    decx::_GPU_Tensor* _A = dynamic_cast<decx::_GPU_Tensor*>(&A);
    decx::_GPU_Tensor* _B = dynamic_cast<decx::_GPU_Tensor*>(&B);

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (_A->width != _B->width || _A->height != _B->height || _A->depth != _B->depth) {
        decx::MDim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::cuda::Dot_fp16_ten(_A, _B, res, &handle, accu_flag);
}