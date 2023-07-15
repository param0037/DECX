/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Vector_fill.h"


_DECX_API_ de::DH de::cpu::Constant_fp32(de::Vector& src, const float value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if (decx::cpu::_get_permitted_concurrency() * 1024 < _src->length / 8) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::bp::fill1D_v256_b32_caller_MT((float*)_src->Vec.ptr, value, _src->length, &t1D);
    }
    else {
        decx::bp::fill1D_v256_b32_caller_ST((float*)_src->Vec.ptr, value, _src->length);
    }

    return handle;
}



_DECX_API_ de::DH de::cpu::Constant_int32(de::Vector& src, const int value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_INT32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if (decx::cpu::_get_permitted_concurrency() * 1024 < _src->length / 8) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::bp::fill1D_v256_b32_caller_MT((float*)_src->Vec.ptr, *((float*)&value), _src->length, &t1D);
    }
    else {
        decx::bp::fill1D_v256_b32_caller_ST((float*)_src->Vec.ptr, *((float*)&value), _src->length);
    }

    return handle;
}




_DECX_API_ de::DH de::cpu::Constant_fp64(de::Vector& src, const double value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_FP64_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if (decx::cpu::_get_permitted_concurrency() * 1024 < _src->length / 4) {
        decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
        decx::bp::fill1D_v256_b64_caller_MT((double*)_src->Vec.ptr, *((double*)&value), _src->length, &t1D);
    }
    else {
        decx::bp::fill1D_v256_b64_caller_ST((double*)_src->Vec.ptr, *((double*)&value), _src->length);
    }

    return handle;
}