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


_DECX_API_ de::DH de::cuda::Integral(de::Vector& src, de::Vector& dst, const int scan_mode)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::cuda_vector_integral_fp32(_src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::calc::cuda_vector_integral_uc8(_src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::cuda_vector_integral_fp16(_src, _dst, scan_mode);
        break;
    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Integral_Async(de::Vector& src, de::Vector& dst, const int scan_mode, de::DecxStream& S)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::async::register_async_task(S.Get_ID(), decx::calc::cuda_vector_integral_fp32,
            _src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::async::register_async_task(S.Get_ID(), decx::calc::cuda_vector_integral_uc8, 
            _src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::async::register_async_task(S.Get_ID(), decx::calc::cuda_vector_integral_fp16, 
            _src, _dst, scan_mode);
        break;
    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Integral(de::GPU_Vector& src, de::GPU_Vector& dst, const int scan_mode)
{
    de::DH handle;

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::calc::cuda_GPU_vector_integral_fp32(_src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::calc::cuda_GPU_vector_integral_uc8(_src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::calc::cuda_GPU_vector_integral_fp16(_src, _dst, scan_mode);
        break;
    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Integral_Async(de::GPU_Vector& src, de::GPU_Vector& dst, const int scan_mode, de::DecxStream& S)
{
    de::DH handle;

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::async::register_async_task(S.Get_ID(), decx::calc::cuda_GPU_vector_integral_fp32, 
            _src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::async::register_async_task(S.Get_ID(), decx::calc::cuda_GPU_vector_integral_uc8, 
            _src, _dst, scan_mode);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::async::register_async_task(S.Get_ID(), decx::calc::cuda_GPU_vector_integral_fp16, 
            _src, _dst, scan_mode);
        break;
    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}