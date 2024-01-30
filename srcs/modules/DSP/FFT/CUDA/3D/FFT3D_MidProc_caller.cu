/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
