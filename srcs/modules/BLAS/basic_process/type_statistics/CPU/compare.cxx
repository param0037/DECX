/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "compare.h"


de::DH de::cpu::Max(de::Matrix& src, void* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }
    if (res == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::bp::_maximum_2D_caller<decx::bp::CPUK::_cmp_kernel_fp32_2D, float, 8>(
            decx::bp::CPUK::_maximum_vec8_fp32_2D, (float*)_src->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), (float*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::bp::_maximum_2D_caller<decx::bp::CPUK::_cmp_kernel_fp64_2D, double, 4>(
            decx::bp::CPUK::_maximum_vec4_fp64_2D, (double*)_src->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), (double*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::bp::_maximum_2D_caller<decx::bp::CPUK::_cmp_kernel_uint8_2D, uint8_t, 16>(
            decx::bp::CPUK::_maximum_vec16_uint8_2D, (uint8_t*)_src->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), (uint8_t*)res);
        break;
    default:
        break;
    }
    
    decx::err::Success(&handle);
    return handle;
}




de::DH de::cpu::Max(de::Vector& src, void* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }
    if (res == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::bp::_maximum_1D_caller<decx::bp::CPUK::_cmp_kernel_fp32_1D, float, 8>(
            decx::bp::CPUK::_maximum_vec8_fp32_1D, (float*)_src->Vec.ptr, _src->length, (float*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::bp::_maximum_1D_caller<decx::bp::CPUK::_cmp_kernel_fp64_1D, double, 4>(
            decx::bp::CPUK::_maximum_vec4_fp64_1D, (double*)_src->Vec.ptr, _src->length, (double*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::bp::_maximum_1D_caller<decx::bp::CPUK::_cmp_kernel_uint8_1D, uint8_t, 8>(
            decx::bp::CPUK::_maximum_vec16_uint8_1D, (uint8_t*)_src->Vec.ptr, _src->length, (uint8_t*)res);
        break;

    default:
        break;
    }
    
    decx::err::Success(&handle);
    return handle;
}




de::DH de::cpu::Min(de::Matrix& src, void* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }
    if (res == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::bp::_minimum_2D_caller<decx::bp::CPUK::_cmp_kernel_fp32_2D, float, 8>(
            decx::bp::CPUK::_minimum_vec8_fp32_2D, (float*)_src->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), (float*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::bp::_minimum_2D_caller<decx::bp::CPUK::_cmp_kernel_fp64_2D, double, 4>(
            decx::bp::CPUK::_minimum_vec4_fp64_2D, (double*)_src->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), (double*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::bp::_minimum_2D_caller<decx::bp::CPUK::_cmp_kernel_uint8_2D, uint8_t, 16>(
            decx::bp::CPUK::_minimum_vec16_uint8_2D, (uint8_t*)_src->Mat.ptr,
            make_uint2(_src->Width(), _src->Height()), _src->Pitch(), (uint8_t*)res);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}




de::DH de::cpu::Min(de::Vector& src, void* res)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }
    if (res == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
            INVALID_PARAM);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::bp::_minimum_1D_caller<decx::bp::CPUK::_cmp_kernel_fp32_1D, float, 8>(
            decx::bp::CPUK::_minimum_vec8_fp32_1D, (float*)_src->Vec.ptr, _src->length, (float*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::bp::_minimum_1D_caller<decx::bp::CPUK::_cmp_kernel_fp64_1D, double, 4>(
            decx::bp::CPUK::_minimum_vec4_fp64_1D, (double*)_src->Vec.ptr, _src->length, (double*)res);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::bp::_minimum_1D_caller<decx::bp::CPUK::_cmp_kernel_uint8_1D, uint8_t, 8>(
            decx::bp::CPUK::_minimum_vec16_uint8_1D, (uint8_t*)_src->Vec.ptr, _src->length, (uint8_t*)res);
        break;

    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}