/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "2D/FFT2D_kernels.cuh"
#include "3D/FFT3D_planner.cuh"
#include "2D/FFT2D_1way_kernel_callers.cuh"
#include "3D/FFT3D_MidProc_caller.cuh"
#include "CUDA_FFTs.cuh"
#include "../../../core/configs/config.h"


namespace decx
{
namespace dsp {
namespace fft {
    template <typename _type_in> _CRSR_
    static void _FFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, 
        decx::cuda_stream* S, decx::cuda_event* E);


    template <typename _type_in> _CRSR_
    static void _FFT3D_caller_cplxd(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, 
        decx::cuda_stream* S, decx::cuda_event* E);


    template <typename _type_out> _CRSR_
    static void _IFFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, 
        decx::cuda_stream* S, decx::cuda_event* E);


    template <typename _type_out> _CRSR_
    static void _IFFT3D_caller_cplxd(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, 
        decx::cuda_stream* S, decx::cuda_event* E);
}
}
}



template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst,
    decx::cuda_stream* S, decx::cuda_event* E)
{
    if (decx::dsp::fft::cuda_FFT3D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT3D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT3D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT3D_planner<float>::release);
    }

    decx::dsp::fft::cuda_FFT3D_cplxf32_planner.lock();
    decx::dsp::fft::_cuda_FFT3D_planner<float>* _planner =
        decx::dsp::fft::cuda_FFT3D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT3D_planner<float>>();

    if (_planner->changed(&src->get_layout(), &dst->get_layout())) {
        _planner->plan(&src->get_layout(), &dst->get_layout(), de::GetLastError(), S);
        Check_Runtime_Error(de::GetLastError());
    }

    _planner->Forward<_type_in>(src, dst, S);

    decx::dsp::fft::cuda_FFT3D_cplxf32_planner.unlock();
}



template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT3D_caller_cplxd(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst,
    decx::cuda_stream* S, decx::cuda_event* E)
{
    if (decx::dsp::fft::cuda_FFT3D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT3D_cplxd64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT3D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT3D_planner<double>::release);
    }

    decx::dsp::fft::cuda_FFT3D_cplxd64_planner.lock();
    decx::dsp::fft::_cuda_FFT3D_planner<double>* _planner =
        decx::dsp::fft::cuda_FFT3D_cplxd64_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT3D_planner<double>>();

    if (_planner->changed(&src->get_layout(), &dst->get_layout())) {
        _planner->plan(&src->get_layout(), &dst->get_layout(), de::GetLastError(), S);
        Check_Runtime_Error(de::GetLastError());
    }

    _planner->Forward<_type_in>(src, dst, S);

    decx::dsp::fft::cuda_FFT3D_cplxd64_planner.unlock();
}


template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst,
    decx::cuda_stream* S, decx::cuda_event* E)
{
    if (decx::dsp::fft::cuda_IFFT3D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT3D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT3D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT3D_planner<float>::release);
    }

    decx::dsp::fft::cuda_IFFT3D_cplxf32_planner.lock();
    decx::dsp::fft::_cuda_FFT3D_planner<float>* _planner =
        decx::dsp::fft::cuda_IFFT3D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT3D_planner<float>>();

    if (_planner->changed(&src->get_layout(), &dst->get_layout())) {
        _planner->plan(&src->get_layout(), &dst->get_layout(), de::GetLastError(), S);
        Check_Runtime_Error(de::GetLastError());
    }

    _planner->Inverse<_type_out>(src, dst, S);

    decx::dsp::fft::cuda_IFFT3D_cplxf32_planner.unlock();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT3D_caller_cplxd(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst,
    decx::cuda_stream* S, decx::cuda_event* E)
{
    if (decx::dsp::fft::cuda_IFFT3D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT3D_cplxd64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT3D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT3D_planner<double>::release);
    }

    decx::dsp::fft::cuda_IFFT3D_cplxd64_planner.lock();
    decx::dsp::fft::_cuda_FFT3D_planner<double>* _planner =
        decx::dsp::fft::cuda_IFFT3D_cplxd64_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT3D_planner<double>>();

    if (_planner->changed(&src->get_layout(), &dst->get_layout())) {
        _planner->plan(&src->get_layout(), &dst->get_layout(), de::GetLastError(), S);
        Check_Runtime_Error(de::GetLastError());
    }

    _planner->Inverse<_type_out>(src, dst, S);

    decx::dsp::fft::cuda_IFFT3D_cplxd64_planner.unlock();
}



_DECX_API_ void de::dsp::cuda::FFT(de::GPU_Tensor& src, de::GPU_Tensor& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::ResetLastError();

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), 
            decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);
    
    if (!(decx::dsp::fft::validate_type_FFT2D(_src->Type()) && decx::dsp::fft::validate_type_FFT2D(_output_type)))
    {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "FFT2D CUDA only supports float, double, uint8_t, de::CPf and de::CPd input");
        return;
    }

    if ((_src->Type() & 3) == 1) {        // (complex)_Fp32
        _dst->re_construct(de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_, _src->Width(), _src->Height(), _src->Depth(), S);
    }
    else if ((_src->Type() & 3) == 2) {       // (complex)_Fp64
        _dst->re_construct(de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_, _src->Width(), _src->Height(), _src->Depth(), S);
    }
    else {  // If is _UINT8_
        _dst->re_construct(_output_type, _src->Width(), _src->Height(), _src->Depth(), S);
    }

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT3D_caller_cplxf<float>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        if (_output_type == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
            decx::dsp::fft::_FFT3D_caller_cplxf<uint8_t>(_src, _dst, S, E);
        }
        else {
            decx::dsp::fft::_FFT3D_caller_cplxd<uint8_t>(_src, _dst, S, E);
        }
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT3D_caller_cplxf<de::CPf>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_FFT3D_caller_cplxd<double>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_FFT3D_caller_cplxd<de::CPd>(_src, _dst, S, E);
        break;

    default:
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



_DECX_API_ void de::dsp::cuda::IFFT(de::GPU_Tensor& src, de::GPU_Tensor& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(),
            decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);

    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    if (!(decx::dsp::fft::validate_type_FFT2D(_src->Type()) && decx::dsp::fft::validate_type_FFT2D(_output_type))) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "FFT2D CUDA only supports float, double, uint8_t, de::CPf and de::CPd input");
        return;
    }

    if (_output_type != de::_DATA_TYPES_FLAGS_::_UINT8_) // Ensures it's either fp32(cplxf) or fp64(cplxd)
    {
        if (!decx::dsp::fft::check_type_matched_FFT(_src->Type(), _output_type)) {
            decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
                "Conversion between fp32 and fp64 in FFT is not supported");
            return;
        }
    }
    else {
        _dst->re_construct(_output_type, _src->Width(), _src->Height(), _src->Depth(), S);
    }

    switch (_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT3D_caller_cplxf<float>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT3D_caller_cplxf<de::CPf>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::_IFFT3D_caller_cplxd<uint8_t>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_IFFT3D_caller_cplxd<double>(_src, _dst, S, E);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_IFFT3D_caller_cplxd<de::CPd>(_src, _dst, S, E);
        break;

    default:
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}
