/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Matrix_math.h"


_DECX_API_ de::DH de::cpu::Log10(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::ClassNotInit<true>(&handle);
        return handle;
    }

    const uint64_t len = (uint64_t)_src->Pitch() * (uint64_t)_dst->Height();
    decx::cpu::Sqrt_Raw_API<true>((float*)_src->Mat.ptr, (float*)_dst->Mat.ptr, len, _src->Type(), &handle);

    decx::err::Success(&handle);
    return handle;
}