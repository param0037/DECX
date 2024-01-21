/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "summation.h"



de::DH de::cpu::Sum_fp32(de::Vector& src, float* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::bp::_summing_fp32_1D_caller((float*)_src->Vec.ptr, _src->_length, res);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cpu::Sum_fp32(de::Matrix& src, float* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    decx::bp::_summing_fp32_1D_caller((float*)_src->Mat.ptr, (uint64_t)_src->Pitch() * (uint64_t)_src->Height(), res);

    decx::err::Success(&handle);
    return handle;
}




de::DH de::cpu::Sum_fp32(de::Tensor& src, float* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);

    decx::bp::_summing_fp32_1D_caller((float*)_src->Tens.ptr, _src->_element_num, res);

    decx::err::Success(&handle);
    return handle;
}


de::DH de::cpu::Sum_fp64(de::Vector& src, double* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::bp::_summing_fp64_1D_caller((double*)_src->Vec.ptr, _src->_length, res);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cpu::Sum_fp64(de::Matrix& src, double* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    decx::bp::_summing_fp64_1D_caller((double*)_src->Mat.ptr, _src->Pitch() * _src->Height(), res);

    decx::err::Success(&handle);
    return handle;
}




de::DH de::cpu::Sum_fp64(de::Tensor& src, double* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);

    decx::bp::_summing_fp64_1D_caller((double*)_src->Tens.ptr, _src->_element_num, res);

    decx::err::Success(&handle);
    return handle;
}