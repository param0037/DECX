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


#include "../CUDA_FFT1D_planner.cuh"
#include "../FFT1D_1st_kernels_dense.cuh"
#include "../../2D/FFT2D_kernels.cuh"
#include "../../../../../../common/Basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "../FFT1D_kernel_callers.cuh"
#include "../../2D/FFT2D_1way_kernel_callers.cuh"


template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT1D_planner<double>::Forward(decx::_Vector* src, decx::_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<double>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    checkCudaErrors(cudaMemcpyAsync(_double_buffer.get_buffer2<void>(), 
        src->Vec.ptr, 
        src->Len() * sizeof(_type_in),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<_type_in, false>(NULL, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::blas::transpose2D_b16_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_FFT1D_END_(de::CPd)>(&_double_buffer, NULL,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, 
        _double_buffer.get_leading_ptr<void>(), 
        src->Len() * sizeof(de::CPd),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
}

template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Forward<double>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Forward<de::CPd>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;



template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT1D_planner<double>::Forward(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<double>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<_type_in, false>(src->Vec.ptr, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S);

    decx::blas::transpose2D_b16_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_FFT1D_END_(de::CPd)>(&_double_buffer, dst->Vec.ptr,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);
}

template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Forward<double>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Forward<de::CPd>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;




template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT1D_planner<double>::Inverse(decx::_Vector* src, decx::_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<double>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    checkCudaErrors(cudaMemcpyAsync(_double_buffer.get_buffer2<void>(), src->Vec.ptr, src->Len() * sizeof(de::CPd),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<de::CPd, true>(NULL, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S, this->get_signal_length());

    decx::blas::transpose2D_b16_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_IFFT1D_END_(_type_out)>(&_double_buffer, NULL,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _double_buffer.get_leading_ptr<void>(), src->Len() * sizeof(_type_out),
        cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
}

template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Inverse<double>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Inverse<de::CPd>(decx::_Vector*, decx::_Vector*, decx::cuda_stream*) const;


template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT1D_planner<double>::Inverse(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, decx::cuda_stream* S) const
{
    const decx::dsp::fft::_cuda_FFT2D_planner<double>* _formal_FFT2D_ptr = this->get_FFT2D_planner();
    decx::utils::double_buffer_manager _double_buffer(_formal_FFT2D_ptr->get_tmp1_ptr<void>(),
                                                      _formal_FFT2D_ptr->get_tmp2_ptr<void>());

    decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<de::CPd, true>(src->Vec.ptr, &_double_buffer,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongH),
        S, this->get_signal_length());

    decx::blas::transpose2D_b16_for_FFT(_double_buffer.get_leading_ptr<double2>(),
                                     _double_buffer.get_lagging_ptr<double2>(),
                                     make_uint2(this->get_larger_FFT_lengths(0), this->get_larger_FFT_lengths(1)),
                                     _formal_FFT2D_ptr->get_buffer_dims().x,
                                     _formal_FFT2D_ptr->get_buffer_dims().y, S);

    _double_buffer.update_states();

    decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_IFFT1D_END_(_type_out)>(&_double_buffer, dst->Vec.ptr,
        _formal_FFT2D_ptr->get_FFT_info(decx::dsp::fft::_FFT_AlongW),
        S);
}

template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Inverse<double>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT1D_planner<double>::Inverse<de::CPd>(decx::_GPU_Vector*, decx::_GPU_Vector*, decx::cuda_stream*) const;


decx::ResourceHandle decx::dsp::fft::cuda_FFT1D_cplxf64_planner;
decx::ResourceHandle decx::dsp::fft::cuda_IFFT1D_cplxf64_planner;
