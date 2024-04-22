/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "FFT1D_kernel_callers.cuh"


template <typename _type_in, bool _div> void 
decx::dsp::fft::FFT1D_partition_cplxf_1st_caller(const void* src, 
                                                 decx::utils::double_buffer_manager* _double_buffer,
                                                 const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, 
                                                 decx::cuda_stream* S,
                                                 const uint64_t _signal_len_total)
{
    if (std::is_same<_type_in, float>::value){
        decx::dsp::fft::FFT1D_1st_R2C_caller_cplxf_caller(src ? src : _double_buffer->get_buffer2<void>(),
                                                         _double_buffer->get_buffer1<void>(),
                                                         _FFT_info->get_radix(0),   _FFT_info->get_signal_len(),
                                                         _FFT_info->_pitchsrc,      _FFT_info->_pitchtmp, S);
    }
    else if (std::is_same<_type_in, de::CPf>::value) {
        decx::dsp::fft::FFT1D_1st_C2C_caller_cplxf_caller<_div>(src ? src : _double_buffer->get_buffer2<void>(),
                                                                _double_buffer->get_buffer1<void>(),
                                                                _FFT_info->get_radix(0),     _FFT_info->get_signal_len(),
                                                                _signal_len_total,
                                                                _FFT_info->_pitchsrc,    _FFT_info->_pitchtmp, S);
    }
    _double_buffer->reset_buffer1_leading();

    for (uint8_t i = 1; i < _FFT_info->partition_num(); ++i) {
        decx::dsp::fft::FFT2D_C2C_caller_cplxf<false>(_double_buffer->get_leading_ptr<float4>(),
                                               _double_buffer->get_lagging_ptr<float4>(),
                                               _FFT_info->get_radix(i),
                                               _FFT_info->get_kernel_info(i),
                                               _FFT_info->_pitchtmp / 2,
                                               _FFT_info->_pitchtmp / 2, S);
        _double_buffer->update_states();
    }
}

template void decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<float, true>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);

template void decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<float, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);

template void decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<de::CPf, true>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);

template void decx::dsp::fft::FFT1D_partition_cplxf_1st_caller<de::CPf, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);



template <bool _div, bool _conj, typename _type_out>
void decx::dsp::fft::FFT1D_partition_cplxf_end_caller(decx::utils::double_buffer_manager* _double_buffer,
                                                      void* dst, 
                                                      const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, 
                                                      decx::cuda_stream* S)
{
    decx::dsp::fft::FFT2D_1st_C2C_caller_cplxf<false>(_double_buffer->get_leading_ptr<float4>(),
                                                    _double_buffer->get_lagging_ptr<float4>(),
                                                    _FFT_info->get_radix(0),    _FFT_info->get_signal_len(),
                                                    _FFT_info->_pitchsrc / 2,   _FFT_info->_pitchtmp / 2, S);
    _double_buffer->update_states();
    for (uint8_t i = 1; i < _FFT_info->partition_num() - 1; ++i) {
        decx::dsp::fft::FFT2D_C2C_caller_cplxf<false>(_double_buffer->get_leading_ptr<float4>(),
                                               _double_buffer->get_lagging_ptr<float4>(),
                                               _FFT_info->get_radix(i),
                                               _FFT_info->get_kernel_info(i),
                                               _FFT_info->_pitchtmp / 2,
                                               _FFT_info->_pitchtmp / 2, S);
        _double_buffer->update_states();
    }

    // Here I need to consider the case that only execute once
    if (std::is_same<_type_out, de::CPf>::value) {
        decx::dsp::fft::FFT1D_end_C2C_caller_cplxf_caller<_conj>(_double_buffer->get_leading_ptr<void>(),
                                               dst ? dst : _double_buffer->get_lagging_ptr<void>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchdst, S);
    }
    else if (std::is_same<_type_out, float>::value) {
        decx::dsp::fft::IFFT1D_end_C2R_caller_cplxf_caller(_double_buffer->get_leading_ptr<void>(),
                                               dst ? dst : _double_buffer->get_lagging_ptr<void>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchdst, S);
    }
    _double_buffer->update_states();
}

template void decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_FFT1D_END_(de::CPf)>(decx::utils::double_buffer_manager*, void*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_IFFT1D_END_(de::CPf)>(decx::utils::double_buffer_manager*, void*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT1D_partition_cplxf_end_caller<_IFFT1D_END_(float)>(decx::utils::double_buffer_manager*, void*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);


// --------------------------------------------------------- double ---------------------------------------------------------


template <typename _type_in, bool _div> void 
decx::dsp::fft::FFT1D_partition_cplxd_1st_caller(const void* src, 
                                                 decx::utils::double_buffer_manager* _double_buffer,
                                                 const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, 
                                                 decx::cuda_stream* S,
                                                 const uint64_t _signal_len_total)
{
    if (std::is_same<_type_in, double>::value){
        decx::dsp::fft::FFT2D_1st_R2C_caller_cplxd(src ? (double*)src : _double_buffer->get_buffer2<double>(),
                                                         _double_buffer->get_buffer1<double2>(),
                                                         _FFT_info->get_radix(0),   _FFT_info->get_signal_len(),
                                                         _FFT_info->_pitchsrc,      _FFT_info->_pitchtmp, S);
    }
    else if (std::is_same<_type_in, de::CPd>::value) {
        decx::dsp::fft::FFT2D_1st_C2C_caller_cplxd<_div>(src ? (double2*)src : _double_buffer->get_buffer2<double2>(),
                                                         _double_buffer->get_buffer1<double2>(),
                                                         _FFT_info->get_radix(0),     _FFT_info->get_signal_len(),
                                                         _FFT_info->_pitchsrc,    _FFT_info->_pitchtmp, S, _signal_len_total);
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

template void decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<double, true>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);

template void decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<double, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);

template void decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<de::CPd, true>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);

template void decx::dsp::fft::FFT1D_partition_cplxd_1st_caller<de::CPd, false>(const void*, decx::utils::double_buffer_manager*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*, const uint64_t);



template <bool _div, bool _conj, typename _type_out>
void decx::dsp::fft::FFT1D_partition_cplxd_end_caller(decx::utils::double_buffer_manager* _double_buffer, 
                                                      void* dst,
                                                      const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, 
                                                      decx::cuda_stream* S)
{
    decx::dsp::fft::FFT2D_1st_C2C_caller_cplxd<false>(_double_buffer->get_leading_ptr<double2>(),
                                                    _double_buffer->get_lagging_ptr<double2>(),
                                                    _FFT_info->get_radix(0),    _FFT_info->get_signal_len(),
                                                    _FFT_info->_pitchsrc,       _FFT_info->_pitchtmp, S);
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
                                               dst ? (double2*)dst : _double_buffer->get_lagging_ptr<double2>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchdst, S);
    }
    else if (std::is_same<_type_out, double>::value) {
        decx::dsp::fft::FFT2D_end_C2R_caller_cplxd(_double_buffer->get_leading_ptr<double2>(),
                                               dst ? (double*)dst : _double_buffer->get_lagging_ptr<double>(),
                                               _FFT_info->get_radix(_FFT_info->partition_num() - 1),
                                               _FFT_info->get_kernel_info(_FFT_info->partition_num() - 1),
                                               _FFT_info->_pitchtmp,
                                               _FFT_info->_pitchdst, S);
    }
    _double_buffer->update_states();
}

template void decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_FFT1D_END_(de::CPd)>(decx::utils::double_buffer_manager*, void*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_IFFT1D_END_(de::CPd)>(decx::utils::double_buffer_manager*, void*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);

template void decx::dsp::fft::FFT1D_partition_cplxd_end_caller<_IFFT1D_END_(double)>(decx::utils::double_buffer_manager*, void*,
    const decx::dsp::fft::_FFT2D_1way_config*, decx::cuda_stream*);
