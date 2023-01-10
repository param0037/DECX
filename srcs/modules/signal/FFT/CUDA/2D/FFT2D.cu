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
#include "../../../../classes/Matrix.h"
#include "../../../../core/configs/config.h"
#include "FFT2D_sub_functions.h"
#include "IFFT2D_sub_functions.h"


namespace de
{
    namespace signal
    {
        namespace cuda {
            _DECX_API_ de::DH FFT2D_R2C(de::Matrix& src, de::Matrix& dst);


            _DECX_API_ de::DH FFT2D_C2C(de::Matrix& src, de::Matrix& dst);


            _DECX_API_ de::DH IFFT2D_C2C(de::Matrix& src, de::Matrix& dst);


            _DECX_API_ de::DH IFFT2D_C2R(de::Matrix& src, de::Matrix& dst);
        }
    }
}



de::DH de::signal::cuda::FFT2D_R2C(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_FP32_ ||
        _dst->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if ((!decx::signal::check_apart(_src->width)) || (!decx::signal::check_apart(_src->height))) {
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

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(_src->width, &handle);
    config[1].FFT1D_config_gen(_src->height, &handle);

    decx::signal::GPU_FFT2D_R2C_fp32_organizer(_src, _dst, config, &handle, S);
    S->detach();

    return handle;
}



de::DH de::signal::cuda::FFT2D_C2C(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if ((!decx::signal::check_apart(_src->width)) || (!decx::signal::check_apart(_src->height))) {
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

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(_src->width, &handle);
    config[1].FFT1D_config_gen(_src->height, &handle);

    decx::signal::GPU_FFT2D_C2C_fp32_organizer(_src, _dst, config, &handle, S);
    S->detach();

    return handle;
}



de::DH de::signal::cuda::IFFT2D_C2C(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if ((!decx::signal::check_apart(_src->width)) ||
        (!decx::signal::check_apart(_src->height))) {
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

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(_src->width, &handle);
    config[1].FFT1D_config_gen(_src->height, &handle);

    decx::signal::GPU_IFFT2D_C2C_fp32_organizer(_src, _dst, config, &handle, S);
    S->detach();

    return handle;
}



de::DH de::signal::cuda::IFFT2D_C2R(de::Matrix& src, de::Matrix& dst)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::cuP.is_init == false) {
        decx::Not_init(&handle);
        Print_Error_Message(4, CUDA_NOT_INIT);
        return handle;
    }

    if (_src->type != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        _dst->type != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    if ((!decx::signal::check_apart(_src->width)) ||
        (!decx::signal::check_apart(_src->height))) {
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

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(_src->width, &handle);
    config[1].FFT1D_config_gen(_src->height, &handle);

    decx::signal::GPU_IFFT2D_C2R_fp32_organizer(_src, _dst, config, &handle, S);
    S->detach();

    return handle;
}