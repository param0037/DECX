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
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Vector.h"
#include "../../../../core/configs/config.h"
#include "FFT1D_sub_functions.h"
#include "dev_FFT1D_sub_functions.h"
#include "IFFT1D_sub_functions.h"
#include "dev_IFFT1D_sub_functions.h"
#include "../../fft_utils.h"




namespace decx {
    namespace signal {
        void FFT1D_R2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        void FFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        void IFFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        void IFFT1D_C2R(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);
    }
}


namespace decx {
    namespace signal {
        void dev_FFT1D_R2C(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);


        void dev_FFT1D_C2C(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);


        void dev_IFFT1D_C2C(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);


        void dev_IFFT1D_C2R(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);
    }
}


namespace de
{
    namespace signal
    {
        namespace cuda {
            _DECX_API_ de::DH FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


            _DECX_API_ de::DH FFT1D(de::GPU_Vector& src, de::GPU_Vector& dst, const int FFT_flag);


            _DECX_API_ de::DH IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


            _DECX_API_ de::DH IFFT1D(de::GPU_Vector& src, de::GPU_Vector& dst, const int FFT_flag);
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

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_COMPLEX_F32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::GPU_FFT1D_R2C_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::dev_FFT1D_R2C(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_COMPLEX_F32_, src->length);

    decx::signal::dev_GPU_FFT1D_R2C_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::FFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH *handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_COMPLEX_F32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::GPU_FFT1D_C2C_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::dev_FFT1D_C2C(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_COMPLEX_F32_, src->length);

    decx::signal::dev_GPU_FFT1D_C2C_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::IFFT1D_C2C(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_COMPLEX_F32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::GPU_IFFT1D_C2C_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::dev_IFFT1D_C2C(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
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

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_COMPLEX_F32_, src->length);

    decx::signal::dev_GPU_IFFT1D_C2C_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::IFFT1D_C2R(decx::_Vector* src, decx::_Vector* dst, de::DH *handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
        decx::err::FFT_Error_length(handle);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail(handle);
        return;
    }

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_FP32_, src->length, decx::DATA_STORE_TYPE::Page_Locked);

    decx::signal::GPU_IFFT1D_C2R_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}




void decx::signal::dev_IFFT1D_C2R(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    const int src_len = src->length;

    if (!decx::signal::check_apart(src_len)) {
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

    decx::signal::CUDA_FFT_Configs config;
    config.FFT1D_config_gen(src_len, handle);

    dst->re_construct(decx::_FP32_, src->length);

    decx::signal::dev_GPU_IFFT1D_C2R_fp32_organizer(src, dst, &config, handle, S, E);
    S->detach();
    E->detach();
}



_DECX_API_ de::DH de::signal::cuda::FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag)
{
    de::DH handle;

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::FFT_R2C:
        decx::signal::FFT1D_R2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::FFT_C2C:
        decx::signal::FFT1D_C2C(_src, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}




_DECX_API_ de::DH de::signal::cuda::FFT1D(de::GPU_Vector& src, de::GPU_Vector& dst, const int FFT_flag)
{
    de::DH handle;

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::FFT_R2C:
        decx::signal::dev_FFT1D_R2C(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::FFT_C2C:
        decx::signal::dev_FFT1D_C2C(_src, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH de::signal::cuda::IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag)
{
    de::DH handle;

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::IFFT_C2R:
        decx::signal::IFFT1D_C2R(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::IFFT_C2C:
        decx::signal::IFFT1D_C2C(_src, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH de::signal::cuda::IFFT1D(de::GPU_Vector& src, de::GPU_Vector& dst, const int FFT_flag)
{
    de::DH handle;

    if (decx::cuda::_is_CUDA_init() == false) {
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    switch (FFT_flag)
    {
    case de::signal::FFT_flags::IFFT_C2R:
        decx::signal::dev_IFFT1D_C2R(_src, _dst, &handle);
        break;

    case de::signal::FFT_flags::IFFT_C2C:
        decx::signal::dev_IFFT1D_C2C(_src, _dst, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}