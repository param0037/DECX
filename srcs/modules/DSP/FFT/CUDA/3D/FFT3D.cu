/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../../../../classes/GPU_Tensor.h"
#include "../../../../core/basic.h"

#include "../2D/FFT2D_kernels.cuh"
#include "../../../../core/utils/double_buffer.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "FFT3D_planner.cuh"
#include "../2D/FFT2D_1way_kernel_callers.cuh"
#include "FFT3D_MidProc_caller.cuh"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in> _CRSR_
        static void _FFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, de::DH* handle);


        template <typename _type_out> _CRSR_
        static void _IFFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, de::DH* handle);
    }
}
}



template <typename _type_in> _CRSR_
static void decx::dsp::fft::_FFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    decx::dsp::fft::_cuda_FFT3D_planner<float> _planner(make_uint3(src->Depth(), src->Width(), src->Height()));
    _planner.plan(&src->get_layout(), &dst->get_layout(), handle, S);
    Check_Runtime_Error(handle);

    decx::utils::double_buffer_manager double_buffer(_planner.get_tmp1_ptr<void>(), _planner.get_tmp2_ptr<void>());
    double_buffer.reset_buffer1_leading();

    // Along H
    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<_type_in, false>(src->Tens.ptr, &double_buffer, 
        _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT3D_planner<float>::_FFT_AlongH), S);

    // Along W
    const decx::dsp::fft::_cuda_FFT3D_mid_config* _along_W = _planner.get_midFFT_info();
    decx::dsp::fft::FFT3D_cplxf_1st_1way_caller<false>(&double_buffer, _along_W, S);

    // Along D
    const decx::dsp::fft::_FFT2D_1way_config* _along_D = _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT3D_planner<float>::_FFT_AlongD);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(_along_D->_pitchtmp, _along_D->get_signal_len()),
                             _along_W->_1way_FFT_conf._pitchdst, 
                             _along_D->_pitchsrc, 
                             S);
    double_buffer.update_states();
    
    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_FFT2D_END_>(&double_buffer, _along_D, S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             (double2*)dst->Tens.ptr,
                             make_uint2(dst->Depth(), _along_D->_pitchtmp),
                             _along_D->_pitchdst, 
                             dst->get_layout().dpitch, S);

    E->event_record(S);
    E->synchronize();

    _planner.release();
}



template <typename _type_out> _CRSR_
static void decx::dsp::fft::_IFFT3D_caller_cplxf(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);

    decx::dsp::fft::_cuda_FFT3D_planner<float> _planner(make_uint3(src->Depth(), src->Width(), src->Height()));
    _planner.plan(&src->get_layout(), &dst->get_layout(), handle, S);
    Check_Runtime_Error(handle);

    decx::utils::double_buffer_manager double_buffer(_planner.get_tmp1_ptr<void>(), _planner.get_tmp2_ptr<void>());
    double_buffer.reset_buffer1_leading();

    // Along H
    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<de::CPf, true>(src->Tens.ptr, &double_buffer, 
        _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT3D_planner<float>::_FFT_AlongH), S);

    // Along W
    const decx::dsp::fft::_cuda_FFT3D_mid_config* _along_W = _planner.get_midFFT_info();
    decx::dsp::fft::FFT3D_cplxf_1st_1way_caller<true>(&double_buffer, _along_W, S);

    // Along D
    const decx::dsp::fft::_FFT2D_1way_config* _along_D = _planner.get_FFT_info(decx::dsp::fft::_cuda_FFT3D_planner<float>::_FFT_AlongD);
    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(_along_D->_pitchtmp, _along_D->get_signal_len()),
                             _along_W->_1way_FFT_conf._pitchdst, 
                             _along_D->_pitchsrc, 
                             S);
    double_buffer.update_states();
    
    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_IFFT2D_END_(_type_out)>(&double_buffer, _along_D, S);

    if (std::is_same<_type_out, de::CPf>::value){
        decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                                 (double2*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst, 
                                 dst->get_layout().dpitch, S);
    }
    else if (std::is_same<_type_out, float>::value){
        decx::bp::transpose2D_b4(double_buffer.get_leading_ptr<float2>(), 
                                 (float2*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst * 2, 
                                 dst->get_layout().dpitch, S);
    }
    else {
        decx::err::handle_error_info_modify<true>(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, 
            UNSUPPORTED_TYPE);
        return;
    }

    E->event_record(S);
    E->synchronize();

    _planner.release();
}



namespace de
{
namespace dsp {
namespace cuda
{
    _DECX_API_ de::DH FFT(de::GPU_Tensor& src, de::GPU_Tensor& dst);


    _DECX_API_ de::DH IFFT(de::GPU_Tensor& src, de::GPU_Tensor& dst, const de::_DATA_TYPES_FLAGS_ type_out);
}
}
}



_DECX_API_ de::DH de::dsp::cuda::FFT(de::GPU_Tensor& src, de::GPU_Tensor& dst)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify<true>(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_FFT3D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_FFT3D_caller_cplxf<de::CPf>(_src, _dst, &handle);
        break;

    default:
        break;
    }

    return handle;
}



_DECX_API_ de::DH de::dsp::cuda::IFFT(de::GPU_Tensor& src, de::GPU_Tensor& dst, const de::_DATA_TYPES_FLAGS_ type_out)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify<true>(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);

    switch (type_out)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::fft::_IFFT3D_caller_cplxf<float>(_src, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_:
        decx::dsp::fft::_IFFT3D_caller_cplxf<de::CPf>(_src, _dst, &handle);
        break;
    default:
        break;
    }

    return handle;
}