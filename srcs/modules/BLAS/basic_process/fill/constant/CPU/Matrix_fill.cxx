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

#include "Matrix_fill.h"


_DECX_API_ de::DH de::cpu::Constant_fp32(de::Matrix& src, const float value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    if (decx::cpu::_get_permitted_concurrency() < _src->Height()) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::bp::fill2D_v256_b32_caller_MT((float*)_src->Mat.ptr, value, make_uint2(_src->Width(), _src->Height()), _src->Pitch(), &t1D);
    }
    else {
        decx::bp::fill2D_v256_b32_caller_ST((float*)_src->Mat.ptr, value, make_uint2(_src->Width(), _src->Height()), _src->Pitch());
    }

    return handle;
}



_DECX_API_ de::DH de::cpu::Constant_int32(de::Matrix& src, const int value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_INT32_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    if (decx::cpu::_get_permitted_concurrency() < _src->Height()) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::bp::fill2D_v256_b32_caller_MT((float*)_src->Mat.ptr, *((float*)&value), make_uint2(_src->Width(), _src->Height()), _src->Pitch(), &t1D);
    }
    else {
        decx::bp::fill2D_v256_b32_caller_ST((float*)_src->Mat.ptr, *((float*)&value), make_uint2(_src->Width(), _src->Height()), _src->Pitch());
    }

    return handle;
}




_DECX_API_ de::DH de::cpu::Constant_fp64(de::Matrix& src, const double value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() != de::_DATA_TYPES_FLAGS_::_FP64_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return handle;
    }

    if (decx::cpu::_get_permitted_concurrency() < _src->Height()) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::bp::fill2D_v256_b64_caller_MT((double*)_src->Mat.ptr, *((double*)&value), make_uint2(_src->Width(), _src->Height()), _src->Pitch(), &t1D);
    }
    else {
        decx::bp::fill2D_v256_b64_caller_ST((double*)_src->Mat.ptr, *((double*)&value), make_uint2(_src->Width(), _src->Height()), _src->Pitch());
    }

    return handle;
}