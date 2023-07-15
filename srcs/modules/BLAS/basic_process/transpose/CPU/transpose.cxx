/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "transpose.h"


_DECX_API_ de::DH de::cpu::Transpose(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->get_layout()._single_element_size == sizeof(float)) {
        decx::bp::transpose_4x4_caller((const float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, 
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), _dst->Pitch());
    }
    else if (_src->get_layout()._single_element_size == sizeof(double)) {
        decx::bp::transpose_2x2_caller((const double*)_src->Mat.ptr, (double*)_dst->Mat.ptr, 
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), _dst->Pitch());
    }

    return handle;
}