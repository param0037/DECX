/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "zeros.h"


_DECX_API_ void de::dsp::cpu::Zeros(de::Vector& src)
{
    de::ResetLastError();

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return;
    }

    decx::alloc::Memset_H(_src->Vec.block, _src->total_bytes, 0);
}


_DECX_API_ void de::dsp::cpu::Zeros(de::Matrix& src)
{
    de::ResetLastError();
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return;
    }

    decx::alloc::Memset_H(_src->Mat.block, _src->get_total_bytes(), 0);
}


_DECX_API_ void de::dsp::cpu::Zeros(de::Tensor& src)
{
    de::ResetLastError();
    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return;
    }

    decx::alloc::Memset_H(_src->Tens.block, _src->total_bytes, 0);
}
