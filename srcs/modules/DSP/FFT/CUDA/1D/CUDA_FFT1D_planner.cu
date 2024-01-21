/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "CUDA_FFT1D_planner.cuh"


template <typename _type_in>
void decx::dsp::fft::_cuda_FFT1D_planner<_type_in>::plan(const uint64_t signal_length, de::DH* handle, decx::cuda_stream* S)
{
    this->_signal_length = signal_length;
    decx::dsp::fft::_radix_apart<true>(this->_signal_length, &this->_all_radixes);

    this->_plan_group_radixes(handle, S);
}

template void decx::dsp::fft::_cuda_FFT1D_planner<float>::plan(const uint64_t, de::DH*, decx::cuda_stream*);
//template void decx::dsp::fft::_cuda_FFT1D_planner<double>::plan(const uint64_t, de::DH*, decx::cuda_stream*);



// Method 1 : combined
template <typename _type_in>
void decx::dsp::fft::_cuda_FFT1D_planner<_type_in>::_plan_group_radixes(de::DH* handle, decx::cuda_stream* S)
{
    const uint32_t _halved_target = (uint32_t)sqrt(this->_signal_length);
    const uint8_t _alignment = 8 / sizeof(_type_in);

    uint32_t _larger_FFT_size = 1, i = 0;
    while (1)
    {
        const uint32_t _current_radix = this->_all_radixes[i];
        if (_larger_FFT_size * _current_radix >= _halved_target) {
            break;
        }
        _larger_FFT_size *= _current_radix;
        ++i;
    }

    std::vector<uint32_t> _used_index;
    uint32_t _min_exceeded = _larger_FFT_size * this->_all_radixes[i];
    for (uint32_t k = i + 1; k < this->_all_radixes.size(); ++k) 
    {
        const uint32_t _current_radix = this->_all_radixes[k];
        const uint32_t _new_LFFT = _larger_FFT_size * _current_radix;
        if (_new_LFFT > _halved_target) {
            if (_new_LFFT < _min_exceeded) {
                _min_exceeded = _new_LFFT;
            }
        }
        else {
            _larger_FFT_size *= _current_radix;
        }
    }

    this->_large_FFT_lengths[0] = _min_exceeded;
    this->_large_FFT_lengths[1] = this->_signal_length / _min_exceeded;

    this->_FFT2D_layout = decx::dsp::fft::_cuda_FFT2D_planner<_type_in>(
        make_uint2(this->_large_FFT_lengths[1], this->_large_FFT_lengths[0]), handle);
    Check_Runtime_Error(handle);
    this->_FFT2D_layout.plan(this->_large_FFT_lengths[1], this->_large_FFT_lengths[0]);
}

template void decx::dsp::fft::_cuda_FFT1D_planner<float>::_plan_group_radixes(de::DH*, decx::cuda_stream*);
//template void decx::dsp::fft::_cuda_FFT1D_planner<double>::_plan_group_radixes(de::DH*, decx::cuda_stream*);


template <typename _type_in>
uint64_t decx::dsp::fft::_cuda_FFT1D_planner<_type_in>::get_signal_length() const
{
    return this->_signal_length;
}

template uint64_t decx::dsp::fft::_cuda_FFT1D_planner<float>::get_signal_length() const;
//template uint64_t decx::dsp::fft::_cuda_FFT1D_planner<double>::get_signal_length() const;


template <typename _type_in>
const decx::dsp::fft::_cuda_FFT2D_planner<_type_in>* 
decx::dsp::fft::_cuda_FFT1D_planner<_type_in>::get_FFT2D_planner() const
{
    return &this->_FFT2D_layout;
}

template const decx::dsp::fft::_cuda_FFT2D_planner<float>*
decx::dsp::fft::_cuda_FFT1D_planner<float>::get_FFT2D_planner() const;

//template const decx::dsp::fft::_cuda_FFT2D_planner<double>*
//decx::dsp::fft::_cuda_FFT1D_planner<double>::get_FFT2D_planner() const;


template <typename _type_in>
uint32_t decx::dsp::fft::_cuda_FFT1D_planner<_type_in>::get_larger_FFT_lengths(const uint8_t _id) const
{
    return this->_large_FFT_lengths[_id];
}

template uint32_t decx::dsp::fft::_cuda_FFT1D_planner<float>::get_larger_FFT_lengths(const uint8_t) const;
//template uint32_t decx::dsp::fft::_cuda_FFT1D_planner<double>::get_larger_FFT_lengths(const uint8_t) const;
