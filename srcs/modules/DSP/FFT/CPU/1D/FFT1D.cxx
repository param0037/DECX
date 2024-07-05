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


#include "../FFT.h"
#include "CPU_FFT1D_planner.h"
#include "FFT1D_kernels.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in>
        void FFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        template <typename _type_in>
        void FFT1D_caller_cplxd(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        template <typename _type_out>
        void IFFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        template <typename _type_out>
        void IFFT1D_caller_cplxd(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);
    }
}
}


template <typename _type_in>
void decx::dsp::fft::FFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::g_cpu_FFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::g_cpu_FFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<float>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<float>::release_buffers);
    }
    decx::dsp::fft::g_cpu_FFT1D_cplxf32_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<float>* _planner =
        decx::dsp::fft::g_cpu_FFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<float>>();
        
    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, &t1D);
    
    decx::dsp::fft::g_cpu_FFT1D_cplxf32_planner.unlock();
}



template <typename _type_in>
void decx::dsp::fft::FFT1D_caller_cplxd(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());
    
    if (decx::dsp::fft::g_cpu_FFT1D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::g_cpu_FFT1D_cplxd64_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<double>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<double>::release_buffers);
    }
    decx::dsp::fft::g_cpu_FFT1D_cplxd64_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<double>* _planner =
        decx::dsp::fft::g_cpu_FFT1D_cplxd64_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<double>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, &t1D);

    decx::dsp::fft::g_cpu_FFT1D_cplxd64_planner.unlock();
}



template <typename _type_out>
void decx::dsp::fft::IFFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::g_cpu_IFFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::g_cpu_IFFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<float>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<float>::release_buffers);
    }

    decx::dsp::fft::g_cpu_IFFT1D_cplxf32_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<float>* _planner =
        decx::dsp::fft::g_cpu_IFFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<float>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, &t1D);

    decx::dsp::fft::g_cpu_IFFT1D_cplxf32_planner.unlock();
}


template <typename _type_out>
void decx::dsp::fft::IFFT1D_caller_cplxd(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::g_cpu_IFFT1D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::g_cpu_IFFT1D_cplxd64_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<double>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<double>::release_buffers);
    }

    decx::dsp::fft::g_cpu_IFFT1D_cplxd64_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<double>* _planner =
        decx::dsp::fft::g_cpu_IFFT1D_cplxd64_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<double>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, &t1D);

    decx::dsp::fft::g_cpu_IFFT1D_cplxd64_planner.unlock();
}



_DECX_API_ void de::dsp::cpu::FFT(de::Vector& src, de::Vector& dst)
{
    de::ResetLastError();

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init, 
            CPU_NOT_INIT);
        return;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_src->Type()) {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::FFT1D_caller<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::FFT1D_caller<de::CPf>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::FFT1D_caller_cplxd<double>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::FFT1D_caller_cplxd<de::CPd>(_src, _dst, de::GetLastError());
        break;

    default:
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_ErrorFlag, 
            MEANINGLESS_FLAG);
        break;
    }
}



_DECX_API_ void de::dsp::cpu::IFFT(de::Vector& src, de::Vector& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::ResetLastError();

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_output_type) {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::IFFT1D_caller<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::IFFT1D_caller<de::CPf>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::IFFT1D_caller_cplxd<double>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::IFFT1D_caller_cplxd<de::CPd>(_src, _dst, de::GetLastError());
        break;

    default:
        decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_ErrorFlag, 
            MEANINGLESS_FLAG);
        break;
    }
}
