/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cpu_dot_fp32.h"


de::DH de::cpu::Dot(de::Vector& A, de::Vector& B, float* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
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

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);

    if (_A->Width() != _B->Width() || _A->Height() != _B->Height()) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::dot::_dot_fp32_1D_caller((float*)_A->Mat.ptr, (float*)_B->Mat.ptr, (uint64_t)_A->Pitch() * (uint64_t)_A->Height(), res);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cpu::Dot(de::Tensor& A, de::Tensor& B, float* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _A = dynamic_cast<decx::_Tensor*>(&A);
    decx::_Tensor* _B = dynamic_cast<decx::_Tensor*>(&B);

    if (_A->Width() != _B->Width() || _A->Height() != _B->Height() || _A->Depth() != _B->Depth()) {
        decx::err::Mat_Dim_Not_Matching(&handle);
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::dot::_dot_fp32_1D_caller((float*)_A->Tens.ptr, (float*)_B->Tens.ptr, _A->get_layout().dp_x_wp * (uint64_t)_A->Height(), res);

    decx::err::Success(&handle);
    return handle;
}