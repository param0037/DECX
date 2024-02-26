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
#include "CUDA_FFT1D_planner.cuh"


#include "FFT1D_1st_kernels_dense.cuh"
#include "../2D/FFT2D_kernels.cuh"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "FFT1D_kernel_callers.cuh"
#include "../2D/FFT2D_1way_kernel_callers.cuh"


template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT1D_planner<float>::Forward(decx::_Vector* src, decx::_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    checkCudaErrors(cudaMemcpyAsync(_double_buffer.get_buffer2<void>(), 
        src->Vec.ptr, 
        src->Len() * sizeof(_type_in),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<_type_in, false>(NULL, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_FFT1D_END_>(&_double_buffer, NULL,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, 
        _double_buffer.get_leading_ptr<void>(), 
        src->Len() * sizeof(de::CPf),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
}

template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Forward<float>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Forward<de::CPf>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;



template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT1D_planner<float>::Forward(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<_type_in, false>(src->Vec.ptr, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_FFT1D_END_>(&_double_buffer, dst->Vec.ptr,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);
}

template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Forward<float>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Forward<de::CPf>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT1D_planner<float>::Inverse(decx::_Vector* src, decx::_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    checkCudaErrors(cudaMemcpyAsync(_double_buffer.get_buffer2<void>(), src->Vec.ptr, src->Len() * sizeof(de::CPf),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<de::CPf, true>(NULL, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S, this->get_signal_length());

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_IFFT1D_END_(_type_out)>(&_double_buffer, NULL,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _double_buffer.get_leading_ptr<void>(), src->Len() * sizeof(_type_out),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
}

template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Inverse<float>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Inverse<de::CPf>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT1D_planner<float>::Inverse(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<float>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<de::CPf, true>(src->Vec.ptr, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S, this->get_signal_length());

    decx::bp::transpose2D_b8_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_IFFT1D_END_(_type_out)>(&_double_buffer, dst->Vec.ptr,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);
}

template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Inverse<float>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<float>::Inverse<de::CPf>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;


decx::dsp::fft::_cuda_FFT1D_planner<float>* decx::dsp::fft::cuda_FFT1D_cplxf32_planner;
decx::dsp::fft::_cuda_FFT1D_planner<float>* decx::dsp::fft::cuda_IFFT1D_cplxf32_planner;
