/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT2D_1way_kernel_callers.cuh"


template <typename _type_in, bool _div>
void decx::dsp::fft::FFT2D_cplxd_1st_1way_caller(const void* src, 
                                                 decx::utils::double_buffer_manager* _double_buffer,
                                                 const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, 
                                                 decx::cuda_stream* S)
{
    if (std::is_same<_type_in, double>::value){
        decx::dsp::fft::FFT2D_1st_R2C_caller_cplxd((double*)src,                _double_buffer->get_buffer1<double2>(),
                                                   _FFT_info->get_radix(0),     _FFT_info->get_signal_len(),
                                                   _FFT_info->_pitchsrc,        _FFT_info->_pitchtmp, S);
    }
    else if (std::is_same<_type_in, uint8_t>::value) {
        /*decx::dsp::fft::FFT2D_1st_R2C_caller_uc8_cplxf((ushort*)src,            _double_buffer->get_buffer1<float4>(),
                                                   _FFT_info->get_radix(0),     _FFT_info->get_signal_len(),
                                                   _FFT_info->_pitchsrc / 2,    _FFT_info->_pitchtmp / 2, S);*/
    }
    else if (std::is_same<_type_in, de::CPd>::value) {
        decx::dsp::fft::FFT2D_1st_C2C_caller_cplxd<_div>((double2*)src,         _double_buffer->get_buffer1<double2>(),
                                                   _FFT_info->get_radix(0),     _FFT_info->get_signal_len(),
                                                   _FFT_info->_pitchsrc,        _FFT_info->_pitchtmp, S);
    }
    _double_buffer->reset_buffer1_leading();

    for (uint8_t i = 1; i < _FFT_info->partition_num(); ++i) {
        decx::dsp::fft::FFT2D_C2C_caller_cplxd<false>(_double_buffer->get_leading_ptr<double2>(),
                                               _double_buffer->get_lagging_ptr<double2>(),
                                               _FFT_info->get_radix(i),
                                               _FFT_info->get_kernel_info(i),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchtmp, S);
        _double_buffer->update_states();
    }
}


template void decx::dsp::fft::FFT2D_cplxd_1st_1way_caller<double, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT2D_cplxd_1st_1way_caller<uint8_t, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT2D_cplxd_1st_1way_caller<de::CPd, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT2D_cplxd_1st_1way_caller<de::CPd, true>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);



template <bool _div, bool _conj, typename _type_out>
void decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller(decx::utils::double_buffer_manager* _double_buffer,
                                                 const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, 
                                                 decx::cuda_stream* S)
{
    decx::dsp::fft::FFT2D_1st_C2C_caller_cplxd<_div>(_double_buffer->get_leading_ptr<double2>(),
                                                    _double_buffer->get_lagging_ptr<double2>(),
                                                    _FFT_info->get_radix(0),    _FFT_info->get_signal_len(),
                                                    _FFT_info->_pitchsrc,   _FFT_info->_pitchtmp, S);
    _double_buffer->update_states();
    for (uint8_t i = 1; i < _FFT_info->partition_num() - 1; ++i) {
        
        decx::dsp::fft::FFT2D_C2C_caller_cplxd<false>(_double_buffer->get_leading_ptr<double2>(),
                                               _double_buffer->get_lagging_ptr<double2>(),
                                               _FFT_info->get_radix(i),
                                               _FFT_info->get_kernel_info(i),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchtmp, S);
        _double_buffer->update_states();
    }
    // Here I need to consider the case that only execute once
    if (std::is_same<_type_out, de::CPd>::value) {
        decx::dsp::fft::FFT2D_C2C_caller_cplxd<_conj>(_double_buffer->get_leading_ptr<double2>(),
                                               _double_buffer->get_lagging_ptr<double2>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchtmp, S);
    }
    else if (std::is_same<_type_out, uint8_t>::value) {
        //decx::dsp::fft::IFFT2D_C2R_caller_cplxf_u8(_double_buffer->get_leading_ptr<float4>(),
        //                                       _double_buffer->get_lagging_ptr<uchar2>(),
        //                                       _FFT_info->get_radix(_FFT_info->partition_num() - 1),
        //                                       _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
        //                                       _FFT_info->_pitchtmp / 2,        // Times 8 cuz 8 uchars in one de::CPf
        //                                       _FFT_info->_pitchtmp * 8 / 2, S);
    }
    else if (std::is_same<_type_out, double>::value) {
        decx::dsp::fft::FFT2D_end_C2R_caller_cplxd(_double_buffer->get_leading_ptr<double2>(),
                                               _double_buffer->get_lagging_ptr<double>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchtmp * 2, S);    // Times 2 cuz 2 doubles in one de::CPd
    }
    _double_buffer->update_states();
}

template void decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller<_IFFT2D_END_(de::CPd)>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller<_IFFT2D_END_(uint8_t)>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller<_IFFT2D_END_(double)>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT2D_C2C_cplxd_1way_caller<_FFT2D_END_(de::CPd)>(decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);