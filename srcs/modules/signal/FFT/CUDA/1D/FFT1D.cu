/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "../CUDA_FFT_configs.h"
#include "../../../../classes/Vector.h"
#include "../../../../core/configs/config.h"
#include "FFT1D_sub_functions.h"
#include "IFFT1D_sub_functions.h"


namespace de
{
    namespace signal
    {
        namespace cuda {
            _DECX_API_ de::DH FFT1D_R2C(de::Vector& src, de::Vector& dst);


            _DECX_API_ de::DH FFT1D_C2C(de::Vector& src, de::Vector& dst);

        
            _DECX_API_ de::DH IFFT1D_C2C(de::Vector& src, de::Vector& dst);


            _DECX_API_ de::DH IFFT1D_C2R(de::Vector& src, de::Vector& dst);
        }
    }
}




de::DH de::signal::cuda::FFT1D_R2C(de::Vector& src, de::Vector& dst)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle);

    decx::signal::GPU_FFT1D_R2C_fp32_organizer(_src, _dst, &config, &handle, S);
    S->detach();

    return handle;
}



de::DH de::signal::cuda::FFT1D_C2C(de::Vector& src, de::Vector& dst)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle);

    decx::signal::GPU_FFT1D_R2C_fp32_organizer(_src, _dst, &config, &handle, S);
    S->detach();

    return handle;
}



de::DH de::signal::cuda::IFFT1D_C2C(de::Vector& src, de::Vector& dst)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle);

    decx::signal::GPU_IFFT1D_C2C_fp32_organizer(_src, _dst, &config, &handle, S);
    S->detach();
    return handle;
}



de::DH de::signal::cuda::IFFT1D_C2R(de::Vector& src, de::Vector& dst)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        return handle;
    }

    const int src_len = _src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(&handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, &handle);

    decx::signal::GPU_IFFT1D_C2R_fp32_organizer(_src, _dst, &config, &handle, S);
    S->detach();
    return handle;
}