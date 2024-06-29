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


#include "Matrix_integral.h"


#define _SELECTED_CALL_(__func_name, _param1, _param2, _param3) { \
    if (_async_call) {      decx::async::register_async_task(_stream_id, __func_name, _param1, _param2, _param3); }       \
    else {  __func_name(_param1, _param2, _param3); }   \
}


template <bool _async_call>
void decx::scan::Integral2D(decx::_Matrix* src, decx::_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle, const uint32_t _stream_id)
{
    if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_FULL) 
    {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_integral_fp32<false>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_integral_fp16<false>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_integral_uint8_i32<false>(src, dst, scan_calc_mode);
            break;

        default:
            break;
        }
    }
    else if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_VERTICAL) 
    {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_integral_v_fp32(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_integral_v_fp16(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_integral_v_uint8_i32(src, dst, scan_calc_mode);
            break;

        default:
            break;
        }
    }
    else if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_HORIZONTAL)
    {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_integral_fp32<true>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_integral_fp16<true>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_integral_uint8_i32<true>(src, dst, scan_calc_mode);
            break;

        default:
            break;
        }
    }
}


template <bool _async_call>
void decx::scan::dev_Integral2D(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle, const uint32_t _stream_id)
{
    if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_FULL)
    {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_dev_integral_fp32<false>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_dev_integral_fp16<false>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_dev_integral_uint8_i32<false>(src, dst, scan_calc_mode);
            break;

        default:
            break;
        }
    }
    else if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_VERTICAL)
    {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_dev_integral_v_fp32(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_dev_integral_v_fp16(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_dev_integral_v_uint8_i32(src, dst, scan_calc_mode);
            break;

        default:
            break;
        }
    }
    else if (scan2D_mode == decx::scan::SCAN_MODE::SCAN_MODE_HORIZONTAL)
    {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            decx::calc::cuda_matrix_dev_integral_fp32<true>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            decx::calc::cuda_matrix_dev_integral_fp16<true>(src, dst, scan_calc_mode);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            decx::calc::cuda_matrix_dev_integral_uint8_i32<true>(src, dst, scan_calc_mode);
            break;

        default:
            break;
        }
    }
}


_DECX_API_
de::DH de::cuda::Integral(de::Matrix& src, de::Matrix& dst, const int scan2D_mode, const int scan_calc_mode)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }
    
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::scan::Integral2D<false>(_src, _dst, scan2D_mode, scan_calc_mode, &handle);

    return handle;
}


_DECX_API_
de::DH de::cuda::Integral(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int scan2D_mode, const int scan_calc_mode)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::scan::dev_Integral2D<false>(_src, _dst, scan2D_mode, scan_calc_mode, &handle);

    return handle;
}



#ifdef _SELECTED_CALL_
#undef _SELECTED_CALL_
#endif