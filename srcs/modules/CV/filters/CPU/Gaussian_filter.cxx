/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Gaussian_filter.h"



_DECX_API_ void 
de::vis::cpu::Gaussian_Filter(de::Matrix& src, de::Matrix& dst, const de::Point2D neighbor_dims,
    const de::Point2D_f sigmaXY, const int border_type, const bool _is_central, const de::Point2D centerXY)
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
        decx::vis::_Gaussian_filter_uint8_organisor(_src, _dst, make_uint2(neighbor_dims.x, neighbor_dims.y),
            make_float2(sigmaXY.x, sigmaXY.y), make_uint2(neighbor_dims.x / 2, neighbor_dims.y / 2), de::GetLastError(), _is_central, border_type);
        break;

    case de::_DATA_TYPES_FLAGS_::_UCHAR4_:
        decx::vis::_Gaussian_filter_uchar4_organisor(_src, _dst, make_uint2(neighbor_dims.x, neighbor_dims.y),
            make_float2(sigmaXY.x, sigmaXY.y), make_uint2(neighbor_dims.x / 2, neighbor_dims.y / 2), de::GetLastError(), _is_central, border_type);
        break;

    default:
        break;
    }
}