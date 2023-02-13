/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Vector_type_cast.h"



_DECX_API_ de::DH de::cpu::TypeCast(de::Vector& src, de::Vector& dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::DH handle;
    if (!decx::cpI.is_init) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!_src->_init) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        decx::type_cast::_cvtfp32_fp64_caller1D((float*)_src->Vec.ptr, (double*)_dst->Vec.ptr, _src->_length);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        decx::type_cast::_cvtfp64_fp32_caller1D((double*)_src->Vec.ptr, (float*)_dst->Vec.ptr, _src->_length);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP32_INT32) {
        decx::type_cast::_cvtfp32_i32_caller((float*)_src->Vec.ptr, (float*)_dst->Vec.ptr, _src->length);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        decx::type_cast::_cvti32_fp32_caller((float*)_src->Vec.ptr, (float*)_dst->Vec.ptr, _src->length);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        decx::type_cast::_cvtui8_i32_caller1D(
            (float*)_src->Vec.ptr, (float*)_dst->Vec.ptr, _dst->_length / 8);
    }
    else if (cvt_method > 5 && cvt_method < 16) {
        decx::type_cast::_cvti32_ui8_caller1D((float*)_src->Vec.ptr, (int*)_dst->Vec.ptr, _src->_length / 8, cvt_method, &handle);
    }
    else {
        Print_Error_Message(4, MEANINGLESS_FLAG);
        decx::err::InvalidParam(&handle);
        return handle;
    }

    return handle;
}