/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
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