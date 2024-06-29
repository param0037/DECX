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
        decx::dsp::fft::FFT2D_1st_R2C_caller_uc8_cplxd((uint8_t*)src,           _double_buffer->get_buffer1<double2>(),
                                                   _FFT_info->get_radix(0),     _FFT_info->get_signal_len(),
                                                   _FFT_info->_pitchsrc,        _FFT_info->_pitchtmp, S);
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
        decx::dsp::fft::IFFT2D_C2R_caller_cplxd_u8(_double_buffer->get_leading_ptr<double2>(),
                                               _double_buffer->get_lagging_ptr<uint8_t>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchtmp * 16, S);   // Times 16 cuz 8 uchars in one de::CPd
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
