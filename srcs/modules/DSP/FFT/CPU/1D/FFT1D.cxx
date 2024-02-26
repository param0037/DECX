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
#include "CPU_FFT1D_planner.h"
#include "FFT1D_kernels.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in>
        void FFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);


        template <typename _type_out>
        void IFFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle);
    }
}
}


template <typename _type_in>
void decx::dsp::fft::FFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    //decx::dsp::fft::cpu_FFT1D_planner<float> _planner/*(src->Len())*/;
    if (decx::dsp::fft::cpu_FFT1D_cplxf32_planner == NULL) {
        decx::dsp::fft::cpu_FFT1D_cplxf32_planner = new decx::dsp::fft::cpu_FFT1D_planner<float>;
    }
    if (decx::dsp::fft::cpu_FFT1D_cplxf32_planner->changed(src->Len(), t1D.total_thread)) {
        decx::dsp::fft::cpu_FFT1D_cplxf32_planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    /*decx::utils::double_buffer_manager _double_buffer(_planner.get_tmp1_ptr(), 
                                                      _planner.get_tmp2_ptr());
    
    decx::dsp::fft::FIMT1D _FIMT1D(_planner.get_signal_len(), _planner.get_smaller_FFT_info_ptr(0)->get_signal_len());
    
    
    if (_planner.get_kernel_call_num() > 1) {
        _FIMT1D.update(_planner.get_smaller_FFT_info_ptr(1)->get_signal_len());

        decx::dsp::fft::_FFT1D_cplxf32_1st<false, _type_in>((const _type_in*)src->Vec.ptr, 
                                                     (double*)_double_buffer._MIF1.mem, 
                                                     &_planner, &t1D, &_FIMT1D);

        _double_buffer.reset_buffer1_leading();
    }
    else {
        decx::dsp::fft::_FFT1D_cplxf32_1st<false, _type_in>((const _type_in*)src->Vec.ptr, 
                                                     (double*)dst->Vec.ptr, 
                                                     &_planner, &t1D, NULL);
    }

    for (uint32_t i = 1; i < _planner.get_kernel_call_num(); ++i)
    {
        if (i < _planner.get_kernel_call_num() - 1) {
            _FIMT1D.update(_planner.get_smaller_FFT_info_ptr(i + 1)->get_signal_len());

            decx::dsp::fft::_FFT1D_cplxf32_mid<double, false>(_double_buffer.get_leading_ptr<const double>(), 
                                                          _double_buffer.get_lagging_ptr<double>(), 
                                                          &_planner, &t1D, 
                                                          i, &_FIMT1D);
        }
        else {
            decx::dsp::fft::_FFT1D_cplxf32_mid<double, true>(_double_buffer.get_leading_ptr<const double>(), 
                                                         (double*)dst->Vec.ptr, 
                                                         &_planner, &t1D, 
                                                         i, NULL);
        }

        _double_buffer.update_states();
    }*/
    decx::dsp::fft::cpu_FFT1D_cplxf32_planner->Forward<_type_in>(src, dst, &t1D);
    //_planner.release_buffers();
}




template <typename _type_out>
void decx::dsp::fft::IFFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    //decx::dsp::fft::cpu_FFT1D_planner<float> _planner/*(src->Len())*/;
    if (decx::dsp::fft::cpu_IFFT1D_cplxf32_planner == NULL) {
        decx::dsp::fft::cpu_IFFT1D_cplxf32_planner = new decx::dsp::fft::cpu_FFT1D_planner<float>;
    }
    if (decx::dsp::fft::cpu_IFFT1D_cplxf32_planner->changed(src->Len(), t1D.total_thread)) {
        decx::dsp::fft::cpu_IFFT1D_cplxf32_planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    /*decx::utils::double_buffer_manager _double_buffer(_planner.get_tmp1_ptr(), 
                                                      _planner.get_tmp2_ptr());
    
    decx::dsp::fft::FIMT1D _FIMT1D(_planner.get_signal_len(), _planner.get_smaller_FFT_info_ptr(0)->get_signal_len());
    
    if (_planner.get_kernel_call_num() > 1) {
        _FIMT1D.update(_planner.get_smaller_FFT_info_ptr(1)->get_signal_len());

        decx::dsp::fft::_FFT1D_cplxf32_1st<true, double>((const double*)src->Vec.ptr, 
                                                   (double*)_double_buffer._MIF1.mem, 
                                                   &_planner, &t1D, &_FIMT1D);

        _double_buffer.reset_buffer1_leading();
    }
    else {
        decx::dsp::fft::_FFT1D_cplxf32_1st<true, double>((const double*)src->Vec.ptr, 
                                                   (double*)dst->Vec.ptr, 
                                                   &_planner, &t1D, NULL);
    }

    for (uint32_t i = 1; i < _planner.get_kernel_call_num(); ++i)
    {
        if (i < _planner.get_kernel_call_num() - 1) {
            _FIMT1D.update(_planner.get_smaller_FFT_info_ptr(i + 1)->get_signal_len());

            decx::dsp::fft::_FFT1D_cplxf32_mid<double, false>(_double_buffer.get_leading_ptr<const double>(),
                                                          _double_buffer.get_lagging_ptr<double>(), 
                                                          &_planner, &t1D, 
                                                          i, &_FIMT1D);
        }
        else {
            decx::dsp::fft::_FFT1D_cplxf32_mid<_type_out, false>(_double_buffer.get_leading_ptr<const double>(),
                                                         (_type_out*)dst->Vec.ptr,
                                                         &_planner, &t1D, 
                                                         i, NULL);
        }

        _double_buffer.update_states();
    }*/
    decx::dsp::fft::cpu_IFFT1D_cplxf32_planner->Inverse<_type_out>(src, dst, &t1D);
    //_planner.release_buffers();
}



_DECX_API_ de::DH de::dsp::cpu::FFT(de::Vector& src, de::Vector& dst)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init, 
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_src->Type()) {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::FFT1D_caller<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::FFT1D_caller<double>(_src, _dst, &handle);
        break;

    default:
        decx::err::handle_error_info_modify<true, 4>(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, 
            MEANINGLESS_FLAG);
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::dsp::cpu::IFFT(de::Vector& src, de::Vector& dst, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_output_type) {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::IFFT1D_caller<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::IFFT1D_caller<double>(_src, _dst, &handle);
        break;

    default:
        decx::err::handle_error_info_modify<true, 4>(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, 
            MEANINGLESS_FLAG);
        break;
    }

    return handle;
}



void decx::dsp::InitFFT1Resources()
{
    decx::dsp::fft::cpu_FFT1D_cplxf32_planner = NULL;
    decx::dsp::fft::cpu_IFFT1D_cplxf32_planner = NULL;
}



void decx::dsp::FreeFFT1Resources()
{
    if (decx::dsp::fft::cpu_FFT1D_cplxf32_planner != NULL) {
        decx::dsp::fft::cpu_FFT1D_cplxf32_planner->release_buffers();
        delete decx::dsp::fft::cpu_FFT1D_cplxf32_planner;
    }
    if (decx::dsp::fft::cpu_IFFT1D_cplxf32_planner != NULL) {
        decx::dsp::fft::cpu_IFFT1D_cplxf32_planner->release_buffers();
        delete decx::dsp::fft::cpu_IFFT1D_cplxf32_planner;
    }
}
