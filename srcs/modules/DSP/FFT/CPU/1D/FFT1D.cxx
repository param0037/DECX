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

    if (decx::dsp::fft::cpu_FFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cpu_FFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<float>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<float>::release_buffers);
    }
    decx::dsp::fft::cpu_FFT1D_cplxf32_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<float>* _planner =
        decx::dsp::fft::cpu_FFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<float>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, &t1D);

    decx::dsp::fft::cpu_FFT1D_cplxf32_planner.unlock();
}



template <typename _type_in>
void decx::dsp::fft::FFT1D_caller_cplxd(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());
    
    if (decx::dsp::fft::cpu_FFT1D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::cpu_FFT1D_cplxd64_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<double>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<double>::release_buffers);
    }
    decx::dsp::fft::cpu_FFT1D_cplxd64_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<double>* _planner =
        decx::dsp::fft::cpu_FFT1D_cplxd64_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<double>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Forward<_type_in>(src, dst, &t1D);

    decx::dsp::fft::cpu_FFT1D_cplxd64_planner.unlock();
}



template <typename _type_out>
void decx::dsp::fft::IFFT1D_caller(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::cpu_IFFT1D_cplxf32_planner._res_ptr == NULL) {
        decx::dsp::fft::cpu_IFFT1D_cplxf32_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<float>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<float>::release_buffers);
    }

    decx::dsp::fft::cpu_IFFT1D_cplxf32_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<float>* _planner =
        decx::dsp::fft::cpu_IFFT1D_cplxf32_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<float>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, &t1D);

    decx::dsp::fft::cpu_IFFT1D_cplxf32_planner.unlock();
}




template <typename _type_out>
void decx::dsp::fft::IFFT1D_caller_cplxd(decx::_Vector* src, decx::_Vector* dst, de::DH* handle)
{
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    if (decx::dsp::fft::cpu_IFFT1D_cplxd64_planner._res_ptr == NULL) {
        decx::dsp::fft::cpu_IFFT1D_cplxd64_planner.RegisterResource(new decx::dsp::fft::cpu_FFT1D_planner<double>,
            5, &decx::dsp::fft::cpu_FFT1D_planner<double>::release_buffers);
    }

    decx::dsp::fft::cpu_IFFT1D_cplxd64_planner.lock();

    decx::dsp::fft::cpu_FFT1D_planner<double>* _planner =
        decx::dsp::fft::cpu_IFFT1D_cplxd64_planner.get_resource_raw_ptr<decx::dsp::fft::cpu_FFT1D_planner<double>>();

    if (_planner->changed(src->Len(), t1D.total_thread)) {
        _planner->plan(src->Len(), &t1D, handle);
        Check_Runtime_Error(handle);
    }

    _planner->Inverse<_type_out>(src, dst, &t1D);

    decx::dsp::fft::cpu_IFFT1D_cplxd64_planner.unlock();
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
        decx::dsp::fft::FFT1D_caller<de::CPf>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::FFT1D_caller_cplxd<double>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::FFT1D_caller_cplxd<de::CPd>(_src, _dst, &handle);
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
        decx::dsp::fft::IFFT1D_caller<de::CPf>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::fft::IFFT1D_caller_cplxd<double>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F64_:
        decx::dsp::fft::IFFT1D_caller_cplxd<de::CPd>(_src, _dst, &handle);
        break;

    default:
        decx::err::handle_error_info_modify<true, 4>(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, 
            MEANINGLESS_FLAG);
        break;
    }

    return handle;
}

