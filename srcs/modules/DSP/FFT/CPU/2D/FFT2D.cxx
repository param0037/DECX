/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT.h"
#include "CPU_FFT2D_planner.h"


namespace decx
{
namespace dsp {
    namespace fft 
    {
        template <typename _type_in>
        _CRSR_ static void FFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);


        template <typename _type_out>
        _CRSR_ static void IFFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
    }
}
}


template <typename _type_in>
static void decx::dsp::fft::FFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::cpu_FFT2D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cpu_FFT2D_cplxf32_planner.RegisterResource(new decx::dsp::fft::cpu_FFT2D_planner<float>,
            5, &decx::dsp::fft::cpu_FFT2D_planner<float>::release_buffers);
    }

    decx::dsp::fft::cpu_FFT2D_cplxf32_planner.lock();

    decx::dsp::fft::cpu_FFT2D_planner<float>* _planner =
        decx::dsp::fft::cpu_FFT2D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT2D_planner<float>>();

    if (_planner->changed(&src->get_layout(), &dst->get_layout(), t1D.total_thread)) {
        _planner->plan<_type_in>(&src->get_layout(), &dst->get_layout(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, &t1D);

    decx::dsp::fft::cpu_FFT2D_cplxf32_planner.unlock();
}


template <typename _type_out>
static void decx::dsp::fft::IFFT2D_caller_cplxf(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    
    if (decx::dsp::fft::cpu_IFFT2D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cpu_IFFT2D_cplxf32_planner.RegisterResource(new decx::dsp::fft::cpu_FFT2D_planner<float>,
            5, &decx::dsp::fft::cpu_FFT2D_planner<float>::release_buffers);
    }

    decx::dsp::fft::cpu_IFFT2D_cplxf32_planner.lock();

    decx::dsp::fft::cpu_FFT2D_planner<float>* _planner =
        decx::dsp::fft::cpu_IFFT2D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT2D_planner<float>>();

    if (_planner->changed(&src->get_layout(), &dst->get_layout(), t1D.total_thread)) {
        _planner->plan<_type_out>(&src->get_layout(), &dst->get_layout(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, &t1D);

    decx::dsp::fft::cpu_IFFT2D_cplxf32_planner.unlock();
}


_DECX_API_ void de::dsp::cpu::FFT(de::Matrix& src, de::Matrix& dst)
{
    de::ResetLastError();

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::FFT2D_caller_cplxf<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::FFT2D_caller_cplxf<uint8_t>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::FFT2D_caller_cplxf<de::CPf>(_src, _dst, de::GetLastError());
        break;

    default:
        break;
    }
}


_DECX_API_ void de::dsp::cpu::IFFT(de::Matrix& src, de::Matrix& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::ResetLastError();

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::IFFT2D_caller_cplxf<float>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::IFFT2D_caller_cplxf<uint8_t>(_src, _dst, de::GetLastError());
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::IFFT2D_caller_cplxf<de::CPf>(_src, _dst, de::GetLastError());
        break;

    default:
        break;
    }
}
