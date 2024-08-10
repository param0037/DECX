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


#include "1D/CUDA_FFT1D_planner.cuh"
#include "CUDA_FFTs.cuh"


namespace decx
{
namespace dsp {
    namespace fft 
    {
        template <typename _type_in> _CRSR_ 
        static void _FFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);

        template <typename _type_in> _CRSR_ 
        static void _FFT1D_cplxf64_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);

        template <typename _type_out> _CRSR_ 
        static void _IFFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);

        template <typename _type_out> _CRSR_ 
        static void _IFFT1D_cplxf64_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle);


        template <typename _type_in> _CRSR_
        static void _FFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);

        template <typename _type_in> _CRSR_
        static void _FFT1D_cplxf64(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        template <typename _type_out> _CRSR_
        static void _IFFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);

        template <typename _type_out> _CRSR_
        static void _IFFT1D_cplxf64(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);
    }
}
}


template <typename _type_in> _CRSR_ 
static void decx::dsp::fft::_FFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<float>::release);
    }
    decx::dsp::fft::cuda_FFT1D_cplxf32_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<float>* _planner =
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<float>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_FFT1D_cplxf32_planner.unlock();
}



template <typename _type_in> _CRSR_ 
static void decx::dsp::fft::_FFT1D_cplxf64_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_FFT1D_cplxf64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT1D_cplxf64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<float>::release);
    }
    decx::dsp::fft::cuda_FFT1D_cplxf64_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<double>* _planner =
        decx::dsp::fft::cuda_FFT1D_cplxf64_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<double>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_FFT1D_cplxf64_planner.unlock();
}



template <typename _type_in> _CRSR_ 
static void decx::dsp::fft::_FFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_FFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<float>::release);
    }

    decx::dsp::fft::cuda_FFT1D_cplxf32_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<float>* _planner =
        decx::dsp::fft::cuda_FFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<float>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_FFT1D_cplxf32_planner.unlock();
}



template <typename _type_in> _CRSR_ 
static void decx::dsp::fft::_FFT1D_cplxf64(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }
    
    if (decx::dsp::fft::cuda_FFT1D_cplxf64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_FFT1D_cplxf64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<double>::release);
    }

    decx::dsp::fft::cuda_FFT1D_cplxf64_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<double>* _planner =
        decx::dsp::fft::cuda_FFT1D_cplxf64_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<double>>();
    
    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_FFT1D_cplxf64_planner.unlock();
}



template <typename _type_out> _CRSR_ 
static void decx::dsp::fft::_IFFT1D_cplxf32_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<float>::release);
    }

    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<float>* _planner =
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<float>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.unlock();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT1D_cplxf64_on_GPU(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_IFFT1D_cplxf64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<double>::release);
    }

    decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<double>* _planner =
        decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<double>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.unlock();
}


template <typename _type_out> _CRSR_ 
static void decx::dsp::fft::_IFFT1D_cplxf32(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_IFFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<float>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<float>::release);
    }

    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<float>* _planner =
        decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<float>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_IFFT1D_cplxf32_planner.unlock();
}



template <typename _type_out> _CRSR_ 
static void decx::dsp::fft::_IFFT1D_cplxf64(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (decx::dsp::fft::cuda_IFFT1D_cplxf64_planner._res_ptr == NULL) {
        decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.RegisterResource(new decx::dsp::fft::_cuda_FFT1D_planner<double>,
            5, &decx::dsp::fft::_cuda_FFT1D_planner<double>::release);
    }

    decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.lock();

    decx::dsp::fft::_cuda_FFT1D_planner<double>* _planner =
        decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.get_resource_raw_ptr<decx::dsp::fft::_cuda_FFT1D_planner<double>>();

    if (_planner->changed(src->Len())) {
        _planner->plan(src->Len(), handle, S);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, S);

    E->event_record(S);
    S->synchronize();

    S->detach();
    E->detach();

    decx::dsp::fft::cuda_IFFT1D_cplxf64_planner.unlock();
}



_DECX_API_ void de::dsp::cuda::FFT(de::GPU_Vector& src, de::GPU_Vector& dst)
{
    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT1D_cplxf32_on_GPU<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT1D_cplxf32_on_GPU<de::CPf>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_FFT1D_cplxf64_on_GPU<double>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_FFT1D_cplxf64_on_GPU<de::CPd>(_src, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}



_DECX_API_ void de::dsp::cuda::FFT(de::Vector& src, de::Vector& dst)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT1D_cplxf32<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT1D_cplxf32<de::CPf>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_FFT1D_cplxf64<double>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_FFT1D_cplxf64<de::CPd>(_src, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}




_DECX_API_ void de::dsp::cuda::IFFT(de::GPU_Vector& src, de::GPU_Vector& dst, const de::_DATA_TYPES_FLAGS_ _type_out)
{
    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    switch (_type_out)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT1D_cplxf32_on_GPU<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT1D_cplxf32_on_GPU<de::CPf>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_IFFT1D_cplxf64_on_GPU<double>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_IFFT1D_cplxf64_on_GPU<de::CPd>(_src, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}




_DECX_API_ void de::dsp::cuda::IFFT(de::Vector& src, de::Vector& dst, const de::_DATA_TYPES_FLAGS_ _type_out)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_type_out)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT1D_cplxf32<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT1D_cplxf32<de::CPf>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::_IFFT1D_cplxf64<double>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::_IFFT1D_cplxf64<de::CPd>(_src, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}
