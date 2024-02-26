/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../2D/FFT2D.h"
#include "../../FFT_commons.h"
#include "CPU_FFT3D_planner.h"
#include "FFT3D_kernels.h"


namespace decx
{
namespace dsp {
    namespace fft 
    {
        template <typename _type_in>
        _CRSR_ static void FFT3D_caller_cplxf(decx::_Tensor* src, decx::_Tensor* dst, de::DH* handle);


        template <typename _type_out>
        _CRSR_ static void IFFT3D_caller_cplxf(decx::_Tensor* src, decx::_Tensor* dst, de::DH* handle);
    }
}
}



template <typename _type_in>
static void decx::dsp::fft::FFT3D_caller_cplxf(decx::_Tensor* src, decx::_Tensor* dst, de::DH* handle)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::FFT3D_cplxf32_planner == NULL) {
        decx::dsp::fft::FFT3D_cplxf32_planner = new decx::dsp::fft::cpu_FFT3D_planner<float>;
    }
    if (decx::dsp::fft::FFT3D_cplxf32_planner->changed(&src->get_layout(), &dst->get_layout(), t1D.total_thread)) {
        decx::dsp::fft::FFT3D_cplxf32_planner->plan<double>(&t1D, &src->get_layout(), &dst->get_layout(), handle);
        Check_Runtime_Error(handle);
    }

    decx::dsp::fft::FFT3D_cplxf32_planner->Forward<_type_in>(src, dst);
}



template <typename _type_out>
static void decx::dsp::fft::IFFT3D_caller_cplxf(decx::_Tensor* src, decx::_Tensor* dst, de::DH* handle)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::IFFT3D_cplxf32_planner == NULL) {
        decx::dsp::fft::IFFT3D_cplxf32_planner = new decx::dsp::fft::cpu_FFT3D_planner<float>;
    }
    if (decx::dsp::fft::IFFT3D_cplxf32_planner->changed(&src->get_layout(), &dst->get_layout(), t1D.total_thread)) {
        decx::dsp::fft::IFFT3D_cplxf32_planner->plan<_type_out>(&t1D, &src->get_layout(), &dst->get_layout(), handle);
        Check_Runtime_Error(handle);
    }

    decx::dsp::fft::IFFT3D_cplxf32_planner->Inverse<_type_out>(src, dst);
}



_DECX_API_ de::DH de::dsp::cpu::FFT(de::Tensor& src, de::Tensor& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify<true>(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init, 
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::FFT3D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::FFT3D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::FFT3D_caller_cplxf<double>(_src, _dst, &handle);
        break;

    default:
        decx::err::handle_error_info_modify<true>(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}




_DECX_API_ de::DH de::dsp::cpu::IFFT(de::Tensor& src, de::Tensor& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify<true>(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    switch (_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::IFFT3D_caller_cplxf<double>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::IFFT3D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::fft::IFFT3D_caller_cplxf<uint8_t>(_src, _dst, &handle);
        break;

    default:
        decx::err::handle_error_info_modify<true>(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}



void decx::dsp::InitFFT3Resources()
{
    decx::dsp::fft::FFT3D_cplxf32_planner = NULL;
    decx::dsp::fft::IFFT3D_cplxf32_planner = NULL;
}



void decx::dsp::FreeFFT3Resources()
{
    if (decx::dsp::fft::FFT3D_cplxf32_planner != NULL) {
        decx::dsp::fft::FFT3D_cplxf32_planner->release();
        delete decx::dsp::fft::FFT3D_cplxf32_planner;
    }
    if (decx::dsp::fft::IFFT3D_cplxf32_planner != NULL) {
        decx::dsp::fft::IFFT3D_cplxf32_planner->release();
        delete decx::dsp::fft::IFFT3D_cplxf32_planner;
    }
}
