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

#include "../2D/FFT2D_kernels.cuh"
#include "../../../../core/utils/double_buffer.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "FFT3D_planner.cuh"
#include "../2D/FFT2D_1way_kernel_callers.cuh"
#include "FFT3D_MidProc_caller.cuh"


template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT3D_planner<float>::Forward(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());
    double_buffer.reset_buffer1_leading();

    // Along H
    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<_type_in, false>(src->Tens.ptr, &double_buffer, 
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH), S);

    // Along W
    const decx::dsp::fft::_cuda_FFT3D_mid_config* _along_W = this->get_midFFT_info();
    decx::dsp::fft::FFT3D_cplxf_1st_1way_caller<false>(&double_buffer, _along_W, S);

    // Along D
    const decx::dsp::fft::_FFT2D_1way_config* _along_D = this->get_FFT_info(decx::dsp::fft::_FFT_AlongD);

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
}

template void decx::dsp::fft::_cuda_FFT3D_planner<float>::Forward<float>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<float>::Forward<de::CPf>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<float>::Forward<uint8_t>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT3D_planner<float>::Inverse(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());
    double_buffer.reset_buffer1_leading();

    // Along H
    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<de::CPf, true>(src->Tens.ptr, &double_buffer, 
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH), S);

    // Along W
    const decx::dsp::fft::_cuda_FFT3D_mid_config* _along_W = this->get_midFFT_info();
    decx::dsp::fft::FFT3D_cplxf_1st_1way_caller<true>(&double_buffer, _along_W, S);

    // Along D
    const decx::dsp::fft::_FFT2D_1way_config* _along_D = this->get_FFT_info(decx::dsp::fft::_FFT_AlongD);
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
    else if (std::is_same<_type_out, uint8_t>::value) {
        decx::bp::transpose2D_b1(double_buffer.get_leading_ptr<uint32_t>(), 
                                 (uint32_t*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst * 8, 
                                 dst->get_layout().dpitch, S);
    }
    else if (std::is_same<_type_out, float>::value){
        decx::bp::transpose2D_b4(double_buffer.get_leading_ptr<float2>(), 
                                 (float2*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst * 2, 
                                 dst->get_layout().dpitch, S);
    }
}

template void decx::dsp::fft::_cuda_FFT3D_planner<float>::Inverse<float>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<float>::Inverse<de::CPf>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<float>::Inverse<uint8_t>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;


decx::dsp::fft::_cuda_FFT3D_planner<float>* decx::dsp::fft::cuda_FFT3D_cplxf32_planner;
decx::dsp::fft::_cuda_FFT3D_planner<float>* decx::dsp::fft::cuda_IFFT3D_cplxf32_planner;
