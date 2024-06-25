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
    static void _FFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);


    template <typename _type_out> _CRSR_
    static void _IFFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);


    template <typename _type_in> _CRSR_
    static void _FFT2D_caller_cplxd(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);


    template <typename _type_out> _CRSR_
    static void _IFFT2D_caller_cplxd(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S, decx::cuda_event* E, de::DH* handle);
}
}
}


template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S, 
    decx::cuda_event* E, de::DH* handle)
{
    if (decx::dsp::fft::cuda_FFT2D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT2D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT2D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT2D_planner<float>::release_buffers);
    }

    decx::dsp::fft::cuda_FFT2D_cplxf32_planner.lock();

    decx::dsp::fft::_cuda_FFT2D_planner<float>* _planner =
        decx::dsp::fft::cuda_FFT2D_cplxf32_planner.get_resource_raw_ptr< decx::dsp::fft::_cuda_FFT2D_planner<float>>();

    if (_planner->changed(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch())) {
        _planner->plan(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch(), handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, S);
    
    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_FFT2D_cplxf32_planner.unlock();
}



template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT2D_caller_cplxd(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S,
    decx::cuda_event* E, de::DH* handle)
{
    if (decx::dsp::fft::cuda_FFT2D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT2D_cplxd64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT2D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT2D_planner<double>::release_buffers);
    }

    decx::dsp::fft::cuda_FFT2D_cplxd64_planner.lock();

    decx::dsp::fft::_cuda_FFT2D_planner<double>* _planner =
        decx::dsp::fft::cuda_FFT2D_cplxd64_planner.get_resource_raw_ptr< decx::dsp::fft::_cuda_FFT2D_planner<double>>();

    if (_planner->changed(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch())) {
        _planner->plan(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch(), handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, S);

    decx::dsp::fft::cuda_FFT2D_cplxd64_planner.unlock();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT2D_caller_cplxf(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S,
    decx::cuda_event* E, de::DH* handle)
{
    if (decx::dsp::fft::cuda_IFFT2D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT2D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT2D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT2D_planner<float>::release_buffers);
    }

    decx::dsp::fft::cuda_IFFT2D_cplxf32_planner.lock();

    decx::dsp::fft::_cuda_FFT2D_planner<float>* _planner =
        decx::dsp::fft::cuda_IFFT2D_cplxf32_planner.get_resource_raw_ptr< decx::dsp::fft::_cuda_FFT2D_planner<float>>();

    if (_planner->changed(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch())) {
        _planner->plan(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch(), handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, S);
    
    decx::dsp::fft::cuda_IFFT2D_cplxf32_planner.unlock();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT2D_caller_cplxd(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S,
    decx::cuda_event* E, de::DH* handle)
{
    if (decx::dsp::fft::cuda_IFFT2D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT2D_cplxd64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT2D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT2D_planner<double>::release_buffers);
    }

    decx::dsp::fft::cuda_IFFT2D_cplxd64_planner.lock();

    decx::dsp::fft::_cuda_FFT2D_planner<double>* _planner =
        decx::dsp::fft::cuda_IFFT2D_cplxd64_planner.get_resource_raw_ptr< decx::dsp::fft::_cuda_FFT2D_planner<double>>();

    if (_planner->changed(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch())) {
        _planner->plan(make_uint2(src->Width(), src->Height()), src->Pitch(), dst->Pitch(), handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, S);

    decx::dsp::fft::cuda_IFFT2D_cplxd64_planner.unlock();
}



_DECX_API_ void de::dsp::cuda::FFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ _type_out)
{
    de::ResetLastError();

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    if (!decx::dsp::fft::validate_type_FFT2D(_src->Type())) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "FFT2D CUDA only supports float, double, uint8_t, de::CPf and de::CPd input");
        return;
    }
    
    if (_src->Type() & 3 == 1) {        // Fp32
        _dst->re_construct(de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_, _src->Width(), _src->Height(), S);
    }
    else if (_src->Type() & 3 == 2) {       // Fp64
        _dst->re_construct(de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_, _src->Width(), _src->Height(), S);
    }
    else if (_src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        _dst->re_construct(_type_out, _src->Width(), _src->Height(), S);
    }

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT2D_caller_cplxf<float>(_src, _dst, S, E, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        if (_type_out == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
            decx::dsp::fft::_FFT2D_caller_cplxf<uint8_t>(_src, _dst, S, E, de::GetLastError());
        }
        else {
            decx::dsp::fft::_FFT2D_caller_cplxd<uint8_t>(_src, _dst, S, E, de::GetLastError());
        }
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT2D_caller_cplxf<de::CPf>(_src, _dst, S, E, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_FFT2D_caller_cplxd<double>(_src, _dst, S, E, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_FFT2D_caller_cplxd<de::CPd>(_src, _dst, S, E, de::GetLastError());
        break;

    default:
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



_DECX_API_ void de::dsp::cuda::IFFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ type_out)
{
    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    if (!decx::dsp::fft::validate_type_FFT2D(_src->Type())) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "FFT2D CUDA only supports float, double, uint8_t, de::CPf and de::CPd input");
        return;
    }

    if (decx::dsp::fft::validate_type_FFT2D(type_out)) // Ensures it's either fp32(cplxf) or fp64(cplxd)
    {   
        if (decx::dsp::fft::check_type_matched_FFT(_src->Type(), type_out)) {
            decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
                "Conversion between fp32 and fp64 in FFT is not supported");
            return;
        }
        else {
            _dst->re_construct(type_out, _src->Width(), _src->Height(), S);
        }
    }

    switch (type_out) 
    {
    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<de::CPf>(_src, _dst, S, E, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT2D_caller_cplxf<float>(_src, _dst, S, E, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        if (_src->Type() & 3 == 1) {
            decx::dsp::fft::_IFFT2D_caller_cplxf<uint8_t>(_src, _dst, S, E, de::GetLastError());
        }
        else {
            decx::dsp::fft::_IFFT2D_caller_cplxd<uint8_t>(_src, _dst, S, E, de::GetLastError());
        }
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_IFFT2D_caller_cplxd<double>(_src, _dst, S, E, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_IFFT2D_caller_cplxd<de::CPd>(_src, _dst, S, E, de::GetLastError());
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}

