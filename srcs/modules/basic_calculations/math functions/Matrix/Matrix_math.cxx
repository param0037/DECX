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


#include "Matrix_math.h"


_DECX_API_ de::DH de::cpu::Log10(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Log10_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::cpu::Log2(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Log2_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::cpu::Exp(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Exp_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::cpu::Sin(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Sin_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::cpu::Cos(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Cos_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::cpu::Tan(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Tan_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::cpu::Asin(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Asin_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::cpu::Acos(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Acos_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_ de::DH de::cpu::Atan(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Atan_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::cpu::Sqrt(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Sqrt_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}