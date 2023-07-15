/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "cpl32_extract.h"
#include "../../handles/decx_handles.h"


_DECX_API_ de::DH de::signal::cpu::Module(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    decx::signal::_cpl32_extract_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Pitch() / 4, _src->Height()),
        _src->Pitch(), _dst->Pitch(), decx::signal::CPUK::_module_fp32_ST2D);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::signal::cpu::Angle(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    decx::signal::_cpl32_extract_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Pitch() / 4, _src->Height()),
        _src->Pitch(), _dst->Pitch(), decx::signal::CPUK::_angle_fp32_ST2D);

    decx::err::Success(&handle);
    return handle;
}