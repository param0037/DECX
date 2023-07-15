/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "type_cast.h"


template <bool _print> _DECX_API_
void decx::type_cast::cpu::_type_cast1D_organiser(void* src, void* dst, const size_t proc_len, 
    const int cvt_method, de::DH * handle)
{
    using namespace decx::type_cast;

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        decx::type_cast::_cvtfp32_fp64_caller1D((float*)src, (double*)dst, proc_len);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        decx::type_cast::_cvtfp64_fp32_caller1D((double*)src, (float*)dst, proc_len);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP32_INT32) {
        decx::type_cast::_cvtfp32_i32_caller((float*)src, (float*)dst, proc_len);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        decx::type_cast::_cvti32_fp32_caller((float*)src, (float*)dst, proc_len);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        decx::type_cast::_cvtui8_i32_caller1D((float*)src, (float*)dst, proc_len / 8);
    }
    else if ((cvt_method > 1 && cvt_method < 10) && (cvt_method != CVT_FP32_FP64)) {
        decx::type_cast::_cvti32_ui8_caller1D<_print>((float*)src, (int*)dst, proc_len / 8, cvt_method, handle);
    }
    else if (cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_CYCLIC) || cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_SATURATED)) {
        decx::type_cast::_cvtf32_ui8_caller1D<_print>((float*)src, (int*)dst, proc_len / 8, cvt_method, handle);
    }
    else if (cvt_method == CVT_UINT8_FP32) {
        decx::type_cast::_cvtui8_f32_caller1D((float*)src, (float*)dst, proc_len / 8);
    }
    else {
        decx::err::InvalidParam<_print>(handle);
        return;
    }
}


template _DECX_API_ void decx::type_cast::cpu::_type_cast1D_organiser<true>(void* src, void* dst, const size_t proc_len,
    const int cvt_method, de::DH* handle);

template _DECX_API_ void decx::type_cast::cpu::_type_cast1D_organiser<false>(void* src, void* dst, const size_t proc_len,
    const int cvt_method, de::DH* handle);



template <bool _print> _DECX_API_
void decx::type_cast::cpu::_type_cast2D_organiser(void* src, void* dst, const ulong2 proc_dims, const uint32_t Wsrc,
    const uint32_t Wdst, const int cvt_method, de::DH* handle)
{
    using namespace decx::type_cast;

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        decx::type_cast::_cvtfp32_fp64_caller2D(
            (float*)src, (double*)dst, make_ulong2(proc_dims.x / 4, proc_dims.y), Wsrc, Wdst);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        decx::type_cast::_cvtfp64_fp32_caller2D(
            (double*)src, (float*)dst, make_ulong2(proc_dims.x / 4, proc_dims.y), Wsrc, Wdst);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP32_INT32) {
        decx::type_cast::_cvtfp32_i32_caller((float*)src, (float*)dst, (size_t)proc_dims.x * (size_t)proc_dims.y);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        decx::type_cast::_cvti32_fp32_caller((float*)src, (float*)dst, (size_t)proc_dims.x * (size_t)proc_dims.y);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        decx::type_cast::_cvtui8_i32_caller2D(
            (float*)src, (float*)dst, make_ulong2(proc_dims.x / 8, proc_dims.y), Wsrc / 4, Wdst);
    }
    else if ((cvt_method > 1 && cvt_method < 10) && (cvt_method != CVT_FP32_FP64)) {
        decx::type_cast::_cvti32_ui8_caller2D<_print>((float*)src, (int*)dst, make_ulong2(proc_dims.x / 8, proc_dims.y),
            Wsrc, Wdst / 4, cvt_method, handle);
    }
    else if (cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_CYCLIC) || cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_SATURATED)) {
        decx::type_cast::_cvtf32_ui8_caller2D<_print>((float*)src, (int*)dst, make_ulong2(proc_dims.x / 8, proc_dims.y),
            Wsrc, Wdst / 4, cvt_method, handle);
    }
    else if (cvt_method == CVT_UINT8_FP32) {
        decx::type_cast::_cvtui8_f32_caller2D((float*)src, (float*)dst, make_ulong2(proc_dims.x / 8, proc_dims.y),
            Wsrc / 4, Wdst);
    }
    else {
        decx::err::InvalidParam<_print>(handle);
        return;
    }
}


template _DECX_API_ void decx::type_cast::cpu::_type_cast2D_organiser<true>(void* src, void* dst, const ulong2 proc_dims, const uint32_t Wsrc,
    const uint32_t Wdst, const int cvt_method, de::DH* handle);

template _DECX_API_ void decx::type_cast::cpu::_type_cast2D_organiser<false>(void* src, void* dst, const ulong2 proc_dims, const uint32_t Wsrc,
    const uint32_t Wdst, const int cvt_method, de::DH* handle);



_DECX_API_ de::DH de::cpu::TypeCast(de::Matrix& src, de::Matrix& dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::type_cast::cpu::_type_cast2D_organiser<true>(_src->Mat.ptr, _dst->Mat.ptr,
        make_ulong2(_src->Pitch(), _src->Height()), _src->Pitch(), _dst->Pitch(), cvt_method, &handle);

    return handle;
}



_DECX_API_ de::DH de::cpu::TypeCast(de::Vector& src, de::Vector& dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::type_cast::cpu::_type_cast1D_organiser<true>(_src->Vec.ptr, _dst->Vec.ptr, 
        min(_src->_length, _dst->_length), cvt_method, &handle);

    return handle;
}




_DECX_API_ de::DH de::cpu::TypeCast(de::MatrixArray& src, de::MatrixArray& dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::type_cast::cpu::_type_cast2D_organiser<true>(_src->MatArr.ptr, _dst->MatArr.ptr,
        make_ulong2(_src->Pitch(), _src->Height() * _src->MatrixNumber()), _src->Pitch(), _dst->Pitch(), cvt_method, &handle);

    return handle;
}


//
//_DECX_API_ de::DH de::cpu::TypeCast(de::Tensor& src, de::Tensor& dst, const int cvt_method)
//{
//    using namespace decx::type_cast;
//
//    de::DH handle;
//    if (!decx::cpu::_is_CPU_init()) {
//        Print_Error_Message(4, CPU_NOT_INIT);
//        decx::err::CPU_Not_init(&handle);
//        return handle;
//    }
//
//    decx::err::Success(&handle);
//
//    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
//    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);
//
//    if (!_src->_init) {
//        Print_Error_Message(4, CLASS_NOT_INIT);
//        decx::err::ClassNotInit(&handle);
//        return handle;
//    }
//
//    decx::type_cast::cpu::_type_cast2D_organiser<true>(_src->Tens.ptr, _dst->Tens.ptr,
//        make_ulong2(_src->dp_x_wp, _src->height), _src->dp_x_wp, _dst->dp_x_wp, cvt_method, &handle);
//
//    return handle;
//}