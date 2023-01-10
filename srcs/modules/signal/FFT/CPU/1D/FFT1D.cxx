/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "FFT1D_sub_functions.h"
#include "IFFT1D_sub_functions.h"
#include "../FFT_utils_kernel.h"
#include "../CPU_FFT_configs.h"
#include "../../../../core/configs/config.h"




namespace de
{
namespace signal
{
    namespace cpu
    {
        _DECX_API_ de::DH FFT1D_R2C(de::Vector& src, de::Vector& dst);


        _DECX_API_ de::DH FFT1D_C2C(de::Vector& src, de::Vector& dst);


        _DECX_API_ de::DH IFFT1D_C2C(de::Vector& src, de::Vector& dst);


        _DECX_API_ de::DH IFFT1D_C2R(de::Vector& src, de::Vector& dst);
    }
}
}


_DECX_API_ de::DH de::signal::cpu::FFT1D_R2C(de::Vector& src, de::Vector& dst)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(_src->length, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    decx::signal::FFT1D_R2C_fp32_organizer(_src, _dst, &_conf, &handle);

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::signal::cpu::FFT1D_C2C(de::Vector& src, de::Vector& dst)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(_src->length, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    decx::signal::FFT1D_R2C_fp32_organizer(_src, _dst, &_conf, &handle);

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::signal::cpu::IFFT1D_C2C(de::Vector& src, de::Vector& dst)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(_src->length, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    decx::signal::IFFT1D_C2C_fp32_organizer(_src, _dst, &_conf, &handle);

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH de::signal::cpu::IFFT1D_C2R(de::Vector& src, de::Vector& dst)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    de::DH handle;
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::signal::CPU_FFT_Configs _conf;
    if (!_conf.FFT1D_config_gen(_src->length, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    decx::signal::IFFT1D_C2R_fp32_organizer(_src, _dst, &_conf, &handle);

    decx::err::Success(&handle);
    return handle;
}