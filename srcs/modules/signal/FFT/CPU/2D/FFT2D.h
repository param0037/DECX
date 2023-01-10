/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CPU_FFT2D_H_
#define _CPU_FFT2D_H_

#include "../CPU_FFT_configs.h"
#include "../../../core/configs/config.h"
#include "FFT2D_sub_functions.h"
#include "IFFT2D_sub_functions.h"


namespace de
{
    namespace signal
    {
        namespace cpu {
            _DECX_API_ de::DH FFT2D_R2C(de::Matrix<float>& src, de::Matrix<de::CPf>& dst);


            _DECX_API_ de::DH FFT2D_C2C(de::Matrix<de::CPf>& src, de::Matrix<de::CPf>& dst);


            _DECX_API_ de::DH IFFT2D_C2C(de::Matrix<de::CPf>& src, de::Matrix<de::CPf>& dst);


            _DECX_API_ de::DH IFFT2D_C2R(de::Matrix<de::CPf>& src, de::Matrix<float>& dst);
        }
    }
}


de::DH de::signal::cpu::FFT2D_R2C(de::Matrix<float>& src, de::Matrix<de::CPf>& dst)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix<float>* _src = dynamic_cast<decx::_Matrix<float>*>(&src);
    decx::_Matrix<de::CPf>* _dst = dynamic_cast<decx::_Matrix<de::CPf>*>(&dst);

    decx::signal::CPU_FFT_Configs _conf[2];
    if (!_conf[0].FFT1D_config_gen(_src->width, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    if (!_conf[1].FFT1D_config_gen(_src->height, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    decx::signal::FFT2D_R2C_fp32_organizer(_src, _dst, _conf, &handle);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::signal::cpu::IFFT2D_C2C(de::Matrix<de::CPf>& src, de::Matrix<de::CPf>& dst)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix<de::CPf>* _src = dynamic_cast<decx::_Matrix<de::CPf>*>(&src);
    decx::_Matrix<de::CPf>* _dst = dynamic_cast<decx::_Matrix<de::CPf>*>(&dst);

    decx::signal::CPU_FFT_Configs _conf[2];
    if (!_conf[0].FFT1D_config_gen(_src->width, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    if (!_conf[1].FFT1D_config_gen(_src->height, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    decx::signal::IFFT2D_C2C_fp32_organizer(_src, _dst, _conf, &handle);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::signal::cpu::FFT2D_C2C(de::Matrix<de::CPf>& src, de::Matrix<de::CPf>& dst)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix<de::CPf>* _src = dynamic_cast<decx::_Matrix<de::CPf>*>(&src);
    decx::_Matrix<de::CPf>* _dst = dynamic_cast<decx::_Matrix<de::CPf>*>(&dst);

    decx::signal::CPU_FFT_Configs _conf[2];
    if (!_conf[0].FFT1D_config_gen(_src->width, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    if (!_conf[1].FFT1D_config_gen(_src->height, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    decx::signal::FFT2D_C2C_fp32_organizer(_src, _dst, _conf, &handle);

    decx::err::Success(&handle);
    return handle;
}



de::DH de::signal::cpu::IFFT2D_C2R(de::Matrix<de::CPf>& src, de::Matrix<float>& dst)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix<de::CPf>* _src = dynamic_cast<decx::_Matrix<de::CPf>*>(&src);
    decx::_Matrix<float>* _dst = dynamic_cast<decx::_Matrix<float>*>(&dst);

    decx::signal::CPU_FFT_Configs _conf[2];
    if (!_conf[0].FFT1D_config_gen(_src->width, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    if (!_conf[1].FFT1D_config_gen(_src->height, &handle)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }
    decx::signal::IFFT2D_C2R_fp32_organizer(_src, _dst, _conf, &handle);

    decx::err::Success(&handle);
    return handle;
}


#endif