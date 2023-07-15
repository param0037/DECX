/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "FFT1D_sub_functions.h"
#include "IFFT1D_sub_functions.h"
#include "../FFT_utils_kernel.h"
#include "../CPU_FFT_configs.h"
#include "../../../../core/configs/config.h"
#include "../../fft_utils.h"


namespace decx {
    namespace signal {
        void FFT1D_R2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        void FFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        void IFFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        void IFFT1D_C2R(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);
    }
}


namespace de
{
namespace signal
{
    namespace cpu
    {
        _DECX_API_ de::DH FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


        _DECX_API_ de::DH IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);
    }
}
}



void decx::signal::FFT1D_R2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(src->length, handle)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    dst->re_construct(decx::_COMPLEX_F32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::FFT1D_R2C_fp32_organizer(src, dst, &_conf, handle);
}



void decx::signal::FFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(src->length, handle)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    dst->re_construct(decx::_COMPLEX_F32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::FFT1D_R2C_fp32_organizer(src, dst, &_conf, handle);
}



void decx::signal::IFFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(src->length, handle)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    dst->re_construct(decx::_COMPLEX_F32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::IFFT1D_C2C_fp32_organizer(src, dst, &_conf, handle);
}



void decx::signal::IFFT1D_C2R(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(src->length, handle)) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    dst->re_construct(decx::_FP32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::IFFT1D_C2R_fp32_organizer(src, dst, &_conf, handle);
}



_DECX_API_ de::DH de::signal::cpu::FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag)
{
    de::DH handle;

    if (decx::cpu::_is_CPU_init() == false) {
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::FFT_R2C:
        decx::signal::FFT1D_R2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::FFT_C2C:
        decx::signal::FFT1D_R2C(_src, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::signal::cpu::IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag)
{
    de::DH handle;

    if (decx::cpu::_is_CPU_init() == false) {
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::IFFT_C2R:
        decx::signal::IFFT1D_C2R(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::IFFT_C2C:
        decx::signal::FFT1D_C2C(_src, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}