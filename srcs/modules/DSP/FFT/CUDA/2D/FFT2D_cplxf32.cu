/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT2D_config.cuh"
#include "../../../../core/utils/double_buffer.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "FFT2D_1way_kernel_callers.cuh"


template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<_type_in, false>(src->Mat.ptr, &double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(this->get_buffer_dims().y, this->get_buffer_dims().x),
                             this->get_buffer_dims().x, 
                             this->get_buffer_dims().y, 
                             S);
    double_buffer.update_states();

    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_FFT2D_END_>(&double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             (double2*)dst->Mat.ptr,
                             make_uint2(dst->Width(), dst->Height()),
                             this->get_buffer_dims().y, 
                             dst->Pitch(), S);
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward<float>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward<de::CPf>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Forward<uint8_t>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT2D_cplxf_1st_1way_caller<de::CPf, true>(src->Mat.ptr, &double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(this->get_buffer_dims().y, this->get_buffer_dims().x),
                             this->get_buffer_dims().x, this->get_buffer_dims().y, 
                             S);
    double_buffer.update_states();

    decx::dsp::fft::FFT2D_C2C_cplxf_1way_caller<_IFFT2D_END_(_type_out)>(&double_buffer,
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);
    if (std::is_same<_type_out, de::CPf>::value) {
        decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                                 (double2*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 this->get_buffer_dims().y, 
                                 dst->Pitch(), S);
    }
    else if (std::is_same<_type_out, uint8_t>::value) {
        decx::bp::transpose2D_b1(double_buffer.get_leading_ptr<uint32_t>(), 
                                 (uint32_t*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 this->get_buffer_dims().y * 8,  // Times 8 cuz 8 uchars in one de::CPf
                                 dst->Pitch(), S);
    }
    else if (std::is_same<_type_out, float>::value) {
        decx::bp::transpose2D_b4(double_buffer.get_leading_ptr<float2>(), 
                                 (float2*)dst->Mat.ptr,
                                 make_uint2(dst->Width(), dst->Height()),
                                 this->get_buffer_dims().y * 2,  // Times 2 cuz 2 floats in one de::CPf
                                 dst->Pitch(), S);
    }
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse<float>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse<de::CPf>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT2D_planner<float>::Inverse<uint8_t>(decx::_GPU_Matrix*, decx::_GPU_Matrix*, decx::cuda_stream*) const;


decx::dsp::fft::_cuda_FFT2D_planner<float>* decx::dsp::fft::cuda_FFT2D_cplxf32_planner;
decx::dsp::fft::_cuda_FFT2D_planner<float>* decx::dsp::fft::cuda_IFFT2D_cplxf32_planner;