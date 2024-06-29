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


#include "cpu_dot_fp32.h"


de::DH de::cpu::Dot(de::Vector& A, de::Vector& B, float* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);

    if (_A->length != _B->length) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            MAT_DIM_NOT_MATCH);
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
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);

    if (_A->Width() != _B->Width() || _A->Height() != _B->Height()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            MAT_DIM_NOT_MATCH);
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
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _A = dynamic_cast<decx::_Tensor*>(&A);
    decx::_Tensor* _B = dynamic_cast<decx::_Tensor*>(&B);

    if (_A->Width() != _B->Width() || _A->Height() != _B->Height() || _A->Depth() != _B->Depth()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            MAT_DIM_NOT_MATCH);
        return handle;
    }

    decx::dot::_dot_fp32_1D_caller((float*)_A->Tens.ptr, (float*)_B->Tens.ptr, _A->get_layout().dp_x_wp * (uint64_t)_A->Height(), res);

    decx::err::Success(&handle);
    return handle;
}