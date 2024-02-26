/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../../../../core/basic.h"
#include "FFT2D_config.cuh"
#include "../CUDA_FFTs.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    template <typename _type_in> _CRSR_
    static void _FFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);


    template <typename _type_out> _CRSR_
    static void _IFFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle);
}
}
}


template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    if (decx::dsp::fft::cuda_FFT2D_cplxf32_planner == NULL) {
        decx::dsp::fft::cuda_FFT2D_cplxf32_planner = new decx::dsp::fft::_cuda_FFT2D_planner<float>;
    }
    if (decx::dsp::fft::cuda_FFT2D_cplxf32_planner->changed(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch())) {
        decx::dsp::fft::cuda_FFT2D_cplxf32_planner->plan(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch(), handle);
        Check_Runtime_Error(handle);
    }

    decx::dsp::fft::cuda_FFT2D_cplxf32_planner->Forward<_type_in>(src, dst, S);
    
    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    if (decx::dsp::fft::cuda_IFFT2D_cplxf32_planner == NULL) {
        decx::dsp::fft::cuda_IFFT2D_cplxf32_planner = new decx::dsp::fft::_cuda_FFT2D_planner<float>;
    }
    if (decx::dsp::fft::cuda_IFFT2D_cplxf32_planner->changed(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch())) {
        decx::dsp::fft::cuda_IFFT2D_cplxf32_planner->plan(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch(), handle);
        Check_Runtime_Error(handle);
    }

    decx::dsp::fft::cuda_IFFT2D_cplxf32_planner->Inverse<_type_out>(src, dst, S);
    
    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



_DECX_API_ de::DH de::dsp::cuda::FFT(de::GPU_Matrix& src, de::GPU_Matrix& dst)
{
    de::DH handle;

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT2D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::_FFT2D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT2D_caller_cplxf<de::CPf>(_src, _dst, &handle);
        break;

    default:
        break;
    }
    

    return handle;
}



_DECX_API_ de::DH de::dsp::cuda::IFFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ type_out)
{
    de::DH handle;

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (type_out) 
    {
    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<de::CPf>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;
    }
    
    return handle;
}


void decx::dsp::InitCUDA_FFT2D_Resources()
{
    decx::dsp::fft::cuda_FFT2D_cplxf32_planner = NULL;
    decx::dsp::fft::cuda_IFFT2D_cplxf32_planner = NULL;
}



void decx::dsp::FreeCUDA_FFT2D_Resources()
{
    if (decx::dsp::fft::cuda_FFT2D_cplxf32_planner != NULL) {
        delete decx::dsp::fft::cuda_FFT2D_cplxf32_planner;
    }
    if (decx::dsp::fft::cuda_IFFT2D_cplxf32_planner != NULL) {
        delete decx::dsp::fft::cuda_IFFT2D_cplxf32_planner;
    }
}
