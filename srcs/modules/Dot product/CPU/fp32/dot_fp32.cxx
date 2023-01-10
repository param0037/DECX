/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "cpu_dot_fp32.h"


de::DH de::cpu::Dot(de::Vector& A, de::Vector& B, float* res)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);

    if (_A->length != _B->length) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::dot::_dot_fp32_1D_caller((float*)_A->Vec.ptr, (float*)_B->Vec.ptr, _A->_length, res);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cpu::Dot(de::Matrix& A, de::Matrix& B, float* res)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);

    if (_A->width != _B->width || _A->height != _B->height) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::dot::_dot_fp32_1D_caller((float*)_A->Mat.ptr, (float*)_B->Mat.ptr, _A->_element_num, res);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cpu::Dot(de::Tensor& A, de::Tensor& B, float* res)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _A = dynamic_cast<decx::_Tensor*>(&A);
    decx::_Tensor* _B = dynamic_cast<decx::_Tensor*>(&B);

    if (_A->width != _B->width || _A->height != _B->height || _A->depth != _B->depth) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::dot::_dot_fp32_1D_caller((float*)_A->Tens.ptr, (float*)_B->Tens.ptr, _A->_element_num, res);

    decx::err::Success(&handle);
    return handle;
}