/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Matrix_integral.h"


void decx::scan::Integral2D(decx::_Matrix* src, decx::_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle)
{
    if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_FULL) 
    {
        switch (src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_integral_fp32<true, false>(src, dst, scan_calc_mode, handle);
            break;

        case decx::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_integral_fp16<true, false>(src, dst, scan_calc_mode, handle);
            break;

        case decx::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_integral_uint8_i32<true, false>(src, dst, scan_calc_mode, handle);
            break;

        default:
            break;
        }
    }
    else if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_VERTICAL) 
    {
        switch (src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_integral_v_fp32<true>(src, dst, scan_calc_mode, handle);
            break;

        case decx::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_integral_v_fp16<true>(src, dst, scan_calc_mode, handle);
            break;

        case decx::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_integral_v_uint8_i32<true>(src, dst, scan_calc_mode, handle);
            break;

        default:
            break;
        }
    }
    else if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_HORIZONTAL)
    {
        switch (src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_integral_fp32<true, true>(src, dst, scan_calc_mode, handle);
            break;

        case decx::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_integral_fp16<true, true>(src, dst, scan_calc_mode, handle);
            break;

        case decx::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_integral_uint8_i32<true, true>(src, dst, scan_calc_mode, handle);
            break;

        default:
            break;
        }
    }
}


_DECX_API_
void de::cuda::Integral(de::Matrix& src, de::Matrix& dst, const int scan2D_mode, const int scan_calc_mode)
{
    de::DH handle;
    
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::scan::Integral2D(_src, _dst, scan2D_mode, scan_calc_mode, &handle);
}