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


#include "../../../../../core/basic.h"
#include "../../2D/FFT2D_kernels.cuh"
#include "../../../../../core/utils/double_buffer.h"
#include "../../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "../FFT3D_planner.cuh"
#include "../../2D/FFT2D_1way_kernel_callers.cuh"
#include "../FFT3D_MidProc_caller.cuh"


template <>
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT3D_planner<double>::Forward(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());
    double_buffer.reset_buffer1_leading();

    // Along H
    decx::dsp::fft::FFT2D_cplxd_1st_1way_caller<_type_in, false>(src->Tens.ptr, &double_buffer, 
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH), S);

    // Along W
    const decx::dsp::fft::_cuda_FFT3D_mid_config* _along_W = this->get_midFFT_info();
#if _CUDA_FFT3D_restrict_coalesce_
    if (this->_sync_dpitchdst_needed) {
        checkCudaErrors(cudaMemcpy2DAsync(double_buffer.get_lagging_ptr<void>(),    _along_W->_1way_FFT_conf._pitchsrc * sizeof(de::CPd),
                                          double_buffer.get_leading_ptr<void>(),    src->get_layout().dpitch * sizeof(de::CPd),
                                          this->_signal_dims.x * sizeof(de::CPf),   src->get_layout().wpitch * src->Height(),
                                          cudaMemcpyDeviceToDevice,                 S->get_raw_stream_ref()));
        double_buffer.update_states();
    }
#endif

    decx::dsp::fft::FFT3D_cplxd_1st_1way_caller<false>(&double_buffer, _along_W, S);

    // Along D
    const decx::dsp::fft::_FFT2D_1way_config* _along_D = this->get_FFT_info(decx::dsp::fft::_FFT_AlongD);

    decx::bp::transpose2D_b16(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(_along_D->_pitchtmp, _along_D->get_signal_len()),
                             _along_W->_1way_FFT_conf._pitchdst, 
                             _along_D->_pitchsrc, 
                             S);
    double_buffer.update_states();
    
    decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller<_FFT2D_END_(de::CPd)>(&double_buffer, _along_D, S);

    decx::bp::transpose2D_b16(double_buffer.get_leading_ptr<double2>(), 
                             (double2*)dst->Tens.ptr,
                             make_uint2(dst->Depth(), _along_D->_pitchtmp),
                             _along_D->_pitchdst, 
                             dst->get_layout().dpitch, S);
}

template void decx::dsp::fft::_cuda_FFT3D_planner<double>::Forward<double>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<double>::Forward<de::CPd>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<double>::Forward<uint8_t>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;


template <>
template <typename _type_out>
void decx::dsp::fft::_cuda_FFT3D_planner<double>::Inverse(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, decx::cuda_stream* S) const
{
    decx::utils::double_buffer_manager double_buffer(this->get_tmp1_ptr<void>(), this->get_tmp2_ptr<void>());
    double_buffer.reset_buffer1_leading();

    // Along H
    decx::dsp::fft::FFT2D_cplxd_1st_1way_caller<de::CPd, true>(src->Tens.ptr, &double_buffer, 
        this->get_FFT_info(decx::dsp::fft::_FFT_AlongH), S);

    // Along W
#if _CUDA_FFT3D_restrict_coalesce_
    if (this->_sync_dpitchdst_needed) {
        checkCudaErrors(cudaMemcpy2DAsync(double_buffer.get_lagging_ptr<void>(),    this->_FFT_W._1way_FFT_conf._pitchsrc * sizeof(de::CPd),
                                          double_buffer.get_leading_ptr<void>(),    src->get_layout().dpitch * sizeof(de::CPd),
                                          this->_signal_dims.x * sizeof(de::CPf),   src->get_layout().wpitch * src->Height(),
                                          cudaMemcpyDeviceToDevice,                 S->get_raw_stream_ref()));
        double_buffer.update_states();
    }
#endif

    const decx::dsp::fft::_cuda_FFT3D_mid_config* _along_W = this->get_midFFT_info();
    decx::dsp::fft::FFT3D_cplxd_1st_1way_caller<true>(&double_buffer, _along_W, S);

    // Along D
    const decx::dsp::fft::_FFT2D_1way_config* _along_D = this->get_FFT_info(decx::dsp::fft::_FFT_AlongD);
    decx::bp::transpose2D_b16(double_buffer.get_leading_ptr<double2>(), 
                             double_buffer.get_lagging_ptr<double2>(),
                             make_uint2(_along_D->_pitchtmp, _along_D->get_signal_len()),
                             _along_W->_1way_FFT_conf._pitchdst, 
                             _along_D->_pitchsrc, 
                             S);
    double_buffer.update_states();
    
    decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller<_IFFT2D_END_(_type_out)>(&double_buffer, _along_D, S);

    if (std::is_same<_type_out, de::CPd>::value){
        decx::bp::transpose2D_b16(double_buffer.get_leading_ptr<double2>(), 
                                 (double2*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst, 
                                 dst->get_layout().dpitch, S);
    }
    else if (std::is_same<_type_out, uint8_t>::value) {
        /*decx::bp::transpose2D_b1(double_buffer.get_leading_ptr<uint32_t>(), 
                                 (uint32_t*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst * 8, 
                                 dst->get_layout().dpitch, S);*/
    }
    else if (std::is_same<_type_out, double>::value){
        decx::bp::transpose2D_b8(double_buffer.get_leading_ptr<double2>(), 
                                 (double2*)dst->Tens.ptr,
                                 make_uint2(dst->Depth(), _along_D->_pitchtmp),
                                 _along_D->_pitchdst * 2, 
                                 dst->get_layout().dpitch, S);
    }
}

template void decx::dsp::fft::_cuda_FFT3D_planner<double>::Inverse<double>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<double>::Inverse<de::CPd>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;
template void decx::dsp::fft::_cuda_FFT3D_planner<double>::Inverse<uint8_t>(decx::_GPU_Tensor*, decx::_GPU_Tensor*, decx::cuda_stream*) const;


decx::ResourceHandle decx::dsp::fft::cuda_FFT3D_cplxd64_planner;
decx::ResourceHandle decx::dsp::fft::cuda_IFFT3D_cplxd64_planner;
