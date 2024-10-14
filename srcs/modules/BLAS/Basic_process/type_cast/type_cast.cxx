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

#include "type_cast.h"
#include <Element_wise/common/cpu_element_wise_planner.h>


_DECX_API_
void decx::type_cast::cpu::_type_cast1D_organiser(void* src, void* dst, const uint64_t proc_len, 
    const int cvt_method, de::DH * handle)
{
    using namespace decx::type_cast;
    using namespace de;
    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        decx::type_cast::typecast1D_general_caller<float, double>(&decx::type_cast::CPUK::_v256_cvtps_pd1D, &_planner, 
            (float*)src, (double*)dst, proc_len, &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvtpd_ps1D, &_planner, 
            (double*)src, (float*)dst, proc_len, &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP32_INT32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvtps_i32, &_planner, 
            (float*)src, (int32_t*)dst, proc_len, &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvti32_ps, &_planner, 
            (int32_t*)src, (float*)dst, proc_len, &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvtui8_i32_1D, &_planner, 
            (uint8_t*)src, (int32_t*)dst, proc_len, &t1D);
    }
    else if ((cvt_method > 1 && cvt_method < 10) && (cvt_method != CVT_FP32_FP64)) {
        auto* _kernel_ptr = decx::type_cast::_cvti32_ui8_selector1D(cvt_method);
        decx::type_cast::typecast1D_general_caller(_kernel_ptr, &_planner, 
            (int32_t*)src, (uint8_t*)dst, proc_len, &t1D);
    }
    else if (cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_CYCLIC) || cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_SATURATED)) {
        auto* _kernel_ptr = decx::type_cast::_cvtf32_ui8_selector1D(cvt_method);
        decx::type_cast::typecast1D_general_caller(_kernel_ptr, &_planner, 
            (float*)src, (uint8_t*)dst, proc_len, &t1D);
    }
    else if (cvt_method == CVT_UINT8_FP32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvtui8_f32_1D, &_planner, 
            (uint8_t*)src, (float*)dst, proc_len, &t1D);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return;
    }
}


_DECX_API_
void decx::type_cast::cpu::_type_cast2D_organiser(void* src, void* dst, const uint2 proc_dims, const uint32_t Wsrc,
    const uint32_t Wdst, const int cvt_method, de::DH* handle)
{
    using namespace decx::type_cast;
    using namespace de;

    decx::cpu_ElementWise2D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        decx::type_cast::typecast2D_general_caller(&decx::type_cast::CPUK::_v256_cvtps_pd2D, &_planner, 
            (float*)src, (double*)dst, proc_dims, Wsrc, Wdst, &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        decx::type_cast::typecast2D_general_caller(&decx::type_cast::CPUK::_v256_cvtpd_ps2D, &_planner, 
            (double*)src, (float*)dst, proc_dims, Wsrc, Wdst, &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP32_INT32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvtps_i32, 
            (decx::cpu_ElementWise1D_planner*)(&_planner), 
            (float*)src, (int32_t*)dst, 
            static_cast<uint64_t>(Wsrc) * static_cast<uint64_t>(proc_dims.y), &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        decx::type_cast::typecast1D_general_caller(&decx::type_cast::CPUK::_v256_cvti32_ps, 
            (decx::cpu_ElementWise1D_planner*)(&_planner), 
            (int32_t*)src, (float*)dst, 
            static_cast<uint64_t>(Wsrc) * static_cast<uint64_t>(proc_dims.y), &t1D);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        decx::type_cast::typecast2D_general_caller(&decx::type_cast::CPUK::_v256_cvtui8_i32_2D, &_planner, 
            (uint8_t*)src, (int32_t*)dst, proc_dims, Wsrc, Wdst, &t1D);
    }
    else if ((cvt_method > 1 && cvt_method < 10) && (cvt_method != CVT_FP32_FP64)) {
        auto* _kernel_ptr = decx::type_cast::_cvti32_ui8_selector2D(cvt_method);
        decx::type_cast::typecast2D_general_caller(_kernel_ptr, &_planner, 
            (int32_t*)src, (uint8_t*)dst, proc_dims, Wsrc, Wdst, &t1D);
    }
    else if (cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_CYCLIC) || cvt_method == (CVT_FP32_UINT8 | CVT_UINT8_SATURATED)) {
        auto* _kernel_ptr = decx::type_cast::_cvtf32_ui8_selector2D(cvt_method);
        decx::type_cast::typecast2D_general_caller(_kernel_ptr, &_planner, 
            (float*)src, (uint8_t*)dst, proc_dims, Wsrc, Wdst, &t1D);
    }
    else if (cvt_method == CVT_UINT8_FP32) {
        decx::type_cast::typecast2D_general_caller(&decx::type_cast::CPUK::_v256_cvtui8_f32_2D, &_planner, 
            (uint8_t*)src, (float*)dst, proc_dims, Wsrc, Wdst, &t1D);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return;
    }
}


_DECX_API_ void de::cpu::TypeCast(de::InputVector src, de::OutputVector dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::ResetLastError();
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    const decx::_Vector* _src = dynamic_cast<const decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
    }

    decx::type_cast::cpu::_type_cast1D_organiser(_src->Vec.ptr, _dst->Vec.ptr, 
        min(_src->_length, _dst->_length), cvt_method, de::GetLastError());
}


_DECX_API_ void de::cpu::TypeCast(de::InputMatrix src, de::OutputMatrix dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::ResetLastError();
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
    }

    const decx::_Matrix* _src = dynamic_cast<const decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!_src->is_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT,
            CLASS_NOT_INIT);
    }

    decx::type_cast::cpu::_type_cast2D_organiser(_src->Mat.ptr, _dst->Mat.ptr,
        make_uint2(_src->Width(), _src->Height()), _src->Pitch(), _dst->Pitch(), cvt_method, de::GetLastError());
}
