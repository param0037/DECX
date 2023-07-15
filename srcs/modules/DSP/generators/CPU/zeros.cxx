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


_DECX_API_ de::DH de::gen::cpu::Zeros(de::Vector& src)
{
    de::DH handle;
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::alloc::Memset_H(_src->Vec.block, _src->total_bytes, 0);

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH de::gen::cpu::Zeros(de::Matrix& src)
{
    de::DH handle;
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::alloc::Memset_H(_src->Mat.block, _src->get_total_bytes(), 0);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::gen::cpu::Zeros(de::Tensor& src)
{
    de::DH handle;
    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::alloc::Memset_H(_src->Tens.block, _src->total_bytes, 0);

    decx::err::Success(&handle);
    return handle;
}