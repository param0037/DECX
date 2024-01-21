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


_DECX_API_ de::DH de::dsp::cpu::Module(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->Type() != de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::dsp::_cpl32_extract_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Pitch() / 4, _src->Height()),
        _src->Pitch(), _dst->Pitch(), decx::dsp::CPUK::_module_fp32_ST2D);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::dsp::cpu::Angle(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->Type() != de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    decx::dsp::_cpl32_extract_caller((de::CPf*)_src->Mat.ptr, (float*)_dst->Mat.ptr, make_uint2(_src->Pitch() / 4, _src->Height()),
        _src->Pitch(), _dst->Pitch(), decx::dsp::CPUK::_angle_fp32_ST2D);

    decx::err::Success(&handle);
    return handle;
}