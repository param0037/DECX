/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Matrix_fill.h"


_DECX_API_ de::DH de::cpu::Constant_fp32(de::Matrix& src, const float value)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() != decx::_DATA_TYPES_FLAGS_::_INT32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
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
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() != decx::_DATA_TYPES_FLAGS_::_FP64_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
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