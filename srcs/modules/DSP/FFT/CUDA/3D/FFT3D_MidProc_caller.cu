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


#include "FFT3D_MidProc_caller.cuh"


template <bool _div>
void decx::dsp::fft::FFT3D_cplxf_1st_1way_caller(decx::utils::double_buffer_manager* _double_buffer,
                                                 const decx::dsp::fft::_cuda_FFT3D_mid_config* _FFT_info,
                                                 decx::cuda_stream* S)
{
    decx::dsp::fft::FFT3D_1st_C2C_caller_cplxf<_div>(_double_buffer->get_leading_ptr<float4>(),
                                                        _double_buffer->get_lagging_ptr<float4>(),
                                                        _FFT_info->_1way_FFT_conf.get_radix(0), 
                                                        _FFT_info->_1way_FFT_conf.get_signal_len(),
                                                        make_uint2(_FFT_info->_signal_pitch_src, _FFT_info->_signal_pitch_dst),
                                                        _FFT_info->_1way_FFT_conf._pitchsrc / 2,
                                                        _FFT_info->_1way_FFT_conf._pitchtmp / 2, 
                                                        _FFT_info->_parallel, S);
    _double_buffer->update_states();

    for (uint8_t i = 1; i < _FFT_info->_1way_FFT_conf.partition_num(); ++i) 
    {
        decx::dsp::fft::FFT3D_C2C_caller_cplxf(_double_buffer->get_leading_ptr<float4>(),
                                               _double_buffer->get_lagging_ptr<float4>(),
                                               _FFT_info->_1way_FFT_conf.get_radix(i),
                                               _FFT_info->_1way_FFT_conf.get_kernel_info(i),
                                               _FFT_info->_signal_pitch_dst,
                                               _FFT_info->_1way_FFT_conf._pitchtmp / 2,
                                               _FFT_info->_1way_FFT_conf._pitchtmp / 2, 
                                               _FFT_info->_parallel, S);
        _double_buffer->update_states();
    }
}

template void decx::dsp::fft::FFT3D_cplxf_1st_1way_caller<true>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_cuda_FFT3D_mid_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT3D_cplxf_1st_1way_caller<false>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_cuda_FFT3D_mid_config*, decx::cuda_stream*);



template <bool _div>
void decx::dsp::fft::FFT3D_cplxd_1st_1way_caller(decx::utils::double_buffer_manager* _double_buffer,
                                                 const decx::dsp::fft::_cuda_FFT3D_mid_config* _FFT_info,
                                                 decx::cuda_stream* S)
{
    decx::dsp::fft::FFT3D_1st_C2C_caller_cplxd<_div>(_double_buffer->get_leading_ptr<double2>(),
                                                        _double_buffer->get_lagging_ptr<double2>(),
                                                        _FFT_info->_1way_FFT_conf.get_radix(0), 
                                                        _FFT_info->_1way_FFT_conf.get_signal_len(),
                                                        make_uint2(_FFT_info->_signal_pitch_src, _FFT_info->_signal_pitch_dst),
                                                        _FFT_info->_1way_FFT_conf._pitchsrc,
                                                        _FFT_info->_1way_FFT_conf._pitchtmp, 
                                                        _FFT_info->_parallel, S);
    _double_buffer->update_states();

    for (uint8_t i = 1; i < _FFT_info->_1way_FFT_conf.partition_num(); ++i) 
    {
        decx::dsp::fft::FFT3D_C2C_caller_cplxd(_double_buffer->get_leading_ptr<double2>(),
                                               _double_buffer->get_lagging_ptr<double2>(),
                                               _FFT_info->_1way_FFT_conf.get_radix(i),
                                               _FFT_info->_1way_FFT_conf.get_kernel_info(i),
                                               _FFT_info->_signal_pitch_dst,
                                               _FFT_info->_1way_FFT_conf._pitchtmp,
                                               _FFT_info->_1way_FFT_conf._pitchtmp, 
                                               _FFT_info->_parallel, S);
        _double_buffer->update_states();
    }
}

template void decx::dsp::fft::FFT3D_cplxd_1st_1way_caller<true>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_cuda_FFT3D_mid_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT3D_cplxd_1st_1way_caller<false>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_cuda_FFT3D_mid_config*, decx::cuda_stream*);
