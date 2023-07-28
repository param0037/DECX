/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Vector_integral.h"


_DECX_API_ void de::cuda::Integral(de::Vector& src, de::Vector& dst, const int scan_mode)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return;
    }

    switch (_src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::cuda_integral_fp32<true>(_src, _dst, scan_mode, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::calc::cuda_integral_uc8<true>(_src, _dst, scan_mode, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::cuda_integral_fp16<true>(_src, _dst, scan_mode, &handle);
        break;
    default:
        break;
    }
}



_DECX_API_ void de::cuda::Integral(de::GPU_Vector& src, de::GPU_Vector& dst, const int scan_mode)
{
    de::DH handle;

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return;
    }

    switch (_src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::cuda_GPU_integral_fp32<true>(_src, _dst, scan_mode, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::calc::cuda_GPU_integral_uc8<true>(_src, _dst, scan_mode, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::cuda_GPU_integral_fp16<true>(_src, _dst, scan_mode, &handle);
        break;
    default:
        break;
    }
}