/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "random.h"


_DECX_API_ de::DH 
de::gen::cpu::RandomGaussian(de::Matrix& src, const float mean, const float sigma, de::Point2D_d clipping_range, 
    const uint32_t resolution, const int data_type)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    switch (data_type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _src->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, _src->Width(), _src->Height());
        decx::gen::_gaussian2D_fp32_caller((float*)_src->Mat.ptr, mean, sigma, make_uint2(_src->Width(), _src->Height()),
            _src->Pitch(), make_float2(clipping_range.x, clipping_range.y), resolution);
        break;
    default:
        break;
    }

    decx::err::Success<true>(&handle);
    return handle;
}