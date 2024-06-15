/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "bilateral_filter.h"


_DECX_API_ void 
de::vis::cpu::Bilateral_Filter(de::Matrix& src, de::Matrix& dst, const de::Point2D neighbor_dims,
    const float sigma_space, const float sigma_color, const int border_type)
{
    de::ResetLastError();

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        if (border_type == de::extend_label::_EXTEND_NONE_) 
        {
            _dst->re_construct(_src->Type(), _src->Width() - neighbor_dims.x + 1, _src->Height() - neighbor_dims.y + 1);
            decx::vis::_bilateral_uint8_NB(_src, _dst, make_uint2(neighbor_dims.x, neighbor_dims.y),
                make_float2(sigma_space, sigma_color), de::GetLastError());
        }
        else {
            _dst->re_construct(_src->Type(), _src->Width(), _src->Height());
            decx::vis::_bilateral_uint8_BC(_src, _dst, make_uint2(neighbor_dims.x, neighbor_dims.y),
                make_float2(sigma_space, sigma_color), de::GetLastError(), border_type);
        }
        break;

    case de::_DATA_TYPES_FLAGS_::_UCHAR4_:
        if (border_type == de::extend_label::_EXTEND_NONE_) 
        {
            _dst->re_construct(_src->Type(), _src->Width() - neighbor_dims.x + 1, _src->Height() - neighbor_dims.y + 1);
            decx::vis::_bilateral_uchar4_NB(_src, _dst, make_uint2(neighbor_dims.x, neighbor_dims.y),
                make_float2(sigma_space, sigma_color), de::GetLastError());
        }
        else {
            _dst->re_construct(_src->Type(), _src->Width(), _src->Height());
            decx::vis::_bilateral_uchar4_BC(_src, _dst, make_uint2(neighbor_dims.x, neighbor_dims.y),
                make_float2(sigma_space, sigma_color), de::GetLastError(), border_type);
        }
        break;

    default:
        break;
    }
}