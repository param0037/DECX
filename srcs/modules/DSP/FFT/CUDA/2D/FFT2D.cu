/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../CUDA_FFT_configs.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../core/configs/config.h"
#include "FFT2D_sub_functions.h"
#include "dev_FFT2D_sub_functions.h"
#include "IFFT2D_sub_functions.h"
#include "dev_IFFT2D_sub_functions.h"
#include "../../fft_utils.h"


namespace decx
{
    namespace signal {
        void FFT2D_R2C(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);


        void FFT2D_C2C(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);


        void IFFT2D_C2C(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);


        void IFFT2D_C2R(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
    }
}


namespace decx
{
    namespace signal {
        void dev_FFT2D_R2C(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);


        void dev_FFT2D_C2C(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);


        void dev_IFFT2D_C2C(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);


        void dev_IFFT2D_C2R(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);
    }
}



namespace de
{
    namespace signal
    {
        namespace cuda {
            _DECX_API_ de::DH FFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


            _DECX_API_ de::DH FFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag);


            _DECX_API_ de::DH IFFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


            _DECX_API_ de::DH IFFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag);
        }
    }
}



de::DH de::signal::cuda::FFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag)
{
    de::DH handle;
    
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::FFT_R2C:
        decx::signal::FFT2D_R2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::FFT_C2C:
        decx::signal::FFT2D_C2C(_src, _dst, &handle);
        break;
    }

    return handle;
}




de::DH de::signal::cuda::FFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag)
{
    de::DH handle;
    
    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::FFT_R2C:
        decx::signal::dev_FFT2D_R2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::FFT_C2C:
        decx::signal::dev_FFT2D_C2C(_src, _dst, &handle);
        break;
    }

    return handle;
}




de::DH de::signal::cuda::IFFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::IFFT_C2C:
        decx::signal::IFFT2D_C2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::IFFT_C2R:
        decx::signal::IFFT2D_C2R(_src, _dst, &handle);
        break;
    }

    return handle;
}






de::DH de::signal::cuda::IFFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag)
{
    de::DH handle;

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::IFFT_C2C:
        decx::signal::dev_IFFT2D_C2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::IFFT_C2R:
        decx::signal::dev_IFFT2D_C2R(_src, _dst, &handle);
        break;
    }

    return handle;
}





void decx::signal::FFT2D_R2C(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }
    
    if ((!decx::signal::check_apart(src->Width())) || (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::GPU_FFT2D_R2C_fp32_organizer(src, dst, config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::dev_FFT2D_R2C(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) || (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::dev_GPU_FFT2D_R2C_fp32_organizer(src, dst, config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::FFT2D_C2C(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) || (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::GPU_FFT2D_C2C_fp32_organizer(src, dst, config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::dev_FFT2D_C2C(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) || (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::dev_GPU_FFT2D_C2C_fp32_organizer(src, dst, config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::IFFT2D_C2C(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) ||
        (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }


    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::GPU_IFFT2D_C2C_fp32_organizer(src, dst, config, handle, S);
    S->detach();
}




void decx::signal::dev_IFFT2D_C2C(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) ||
        (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::dev_GPU_IFFT2D_C2C_fp32_organizer(src, dst, config, handle, S);
    S->detach();
}




void decx::signal::IFFT2D_C2R(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) ||
        (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        Print_Error_Message(4, FFT_ERROR_LENGTH);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::GPU_IFFT2D_C2R_fp32_organizer(src, dst, config, handle, S);
    S->detach();
}




void decx::signal::dev_IFFT2D_C2R(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    if (src->Type() != decx::_DATA_TYPES_FLAGS_::_COMPLEX_F32_ ||
        dst->Type() != decx::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::err::TypeError_NotMatch(handle);
        return;
    }

    if ((!decx::signal::check_apart(src->Width())) ||
        (!decx::signal::check_apart(src->Height()))) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config[2];
    config[0].FFT1D_config_gen(src->Width(), handle);
    config[1].FFT1D_config_gen(src->Height(), handle);

    decx::signal::dev_GPU_IFFT2D_C2R_fp32_organizer(src, dst, config, handle, S);
    S->detach();
}