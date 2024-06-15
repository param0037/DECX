/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "extend.h"


_DECX_API_ void de::cpu::Extend(de::Vector& src, de::Vector& dst, const uint32_t left, const uint32_t right,
    const int extend_type, void* val)
{
    de::ResetLastError();

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return;
    }

    _dst->re_construct(_src->type, _src->length + left + right);

    switch (extend_type)
    {
    case de::extend_label::_EXTEND_REFLECT_:
        decx::bp::_extend1D_reflect(_src, _dst, left, right, de::GetLastError());
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        decx::bp::_extend1D_constant(_src, _dst, val, left, right, de::GetLastError());
        break;
    default:
        break;
    }
}


_DECX_API_ void de::cpu::Extend(de::Matrix& src, de::Matrix& dst, const uint32_t left, const uint32_t right,
    const uint32_t top, const uint32_t bottom, const int extend_type, void* val)
{
    de::ResetLastError();

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return;
    }

    _dst->re_construct(_src->Type(), _src->Width() + left + right, _src->Height() + top + bottom);

    const uint4 _ext_param = make_uint4(left, right, top, bottom);

    switch (extend_type)
    {
    case de::extend_label::_EXTEND_REFLECT_:
        decx::bp::_extend2D_reflect(_src, _dst, _ext_param, de::GetLastError());
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        decx::bp::_extend2D_constant(_src, _dst, val, _ext_param, de::GetLastError());
        break;
    default:
        break;
    }
}
