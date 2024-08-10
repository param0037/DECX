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


#include "FFT2D_config.cuh"


template <typename _data_type>
bool decx::dsp::fft::_cuda_FFT2D_planner<_data_type>::changed(const uint2 signal_dims, 
                                                              const uint32_t pitchsrc, 
                                                              const uint32_t pitchdst) const
{
    return (*((uint64_t*)&this->_signal_dims) ^ *((uint64_t*)&signal_dims)) |
        (this->_FFT_V._pitchsrc ^ pitchsrc) |
        (this->_FFT_H._pitchdst ^ pitchdst);
}

template bool decx::dsp::fft::_cuda_FFT2D_planner<float>::changed(const uint2, const uint32_t, const uint32_t) const;
template bool decx::dsp::fft::_cuda_FFT2D_planner<double>::changed(const uint2, const uint32_t, const uint32_t) const;



template <typename _type_in> _CRSR_
void decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::plan(const uint2 signal_dims, 
                                                         const uint32_t pitchsrc, 
                                                         const uint32_t pitchdst, 
                                                         de::DH* handle)
{
    this->_signal_dims = signal_dims;

    constexpr uint8_t _alignment = 128 / sizeof(_type_in);
    this->_buffer_dims = make_uint2(decx::utils::align<uint32_t>(signal_dims.x, _alignment),
        decx::utils::align<uint32_t>(signal_dims.y, _alignment));

    // Allocate buffers in device
    if (decx::alloc::_device_malloc(&this->_tmp1, this->_buffer_dims.x * this->_buffer_dims.y * sizeof(_type_in) * 2) ||
        decx::alloc::_device_malloc(&this->_tmp2, this->_buffer_dims.x * this->_buffer_dims.y * sizeof(_type_in) * 2)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION, DEV_ALLOC_FAIL);
        return;
    }

    this->_FFT_H.plan(this->_signal_dims.x);
    this->_FFT_V.plan(this->_signal_dims.y);

    this->_FFT_V._pitchsrc = pitchsrc;
    this->_FFT_V._pitchtmp = this->_buffer_dims.x;
    this->_FFT_V._pitchdst = this->_buffer_dims.x;

    this->_FFT_H._pitchsrc = this->_buffer_dims.y;
    this->_FFT_H._pitchtmp = this->_buffer_dims.y;
    this->_FFT_H._pitchdst = pitchdst;
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::plan(const uint2, const uint32_t, const uint32_t, de::DH*);
template void decx::dsp::fft::_cuda_FFT2D_planner<double>::plan(const uint2, const uint32_t, const uint32_t, de::DH*);


uint32_t decx::dsp::fft::_FFT2D_1way_config::get_radix(const uint32_t _index) const
{
    return this->_radix[_index];
}


uint32_t decx::dsp::fft::_FFT2D_1way_config::get_signal_len() const
{
    return this->_signal_length;
}


const decx::dsp::fft::FKI_4_2DK* decx::dsp::fft::_FFT2D_1way_config::get_kernel_info(const uint32_t _index) const
{
    return this->_kernrel_infos.get_const_ptr(_index);
}


uint32_t decx::dsp::fft::_FFT2D_1way_config::partition_num() const
{
    return this->_radix.size();
}


void decx::dsp::fft::_FFT2D_1way_config::plan(const uint32_t signal_length)
{
    this->_signal_length = signal_length;
    decx::dsp::fft::_radix_apart<false>(this->_signal_length, &this->_radix);
    this->_kernrel_infos.define_capacity(this->_radix.size());

    uint32_t _store_pitch = 1, _warp_proc_len = 1;
    // Generate FKI
    for (uint8_t i = 0; i < this->_radix.size(); ++i)
    {
        _warp_proc_len *= this->_radix[i];
        this->_kernrel_infos.emplace_back(_store_pitch, _warp_proc_len, this->_signal_length);
        _store_pitch *= this->_radix[i];
    }
}



template <typename _data_type> const decx::dsp::fft::_FFT2D_1way_config*
decx::dsp::fft::_cuda_FFT2D_planner<_data_type>::get_FFT_info(const decx::dsp::fft::FFT_directions _dir) const
{
    switch (_dir)
    {
    case decx::dsp::fft::FFT_directions::_FFT_AlongW:
        return &this->_FFT_H;
        break;

    case decx::dsp::fft::FFT_directions::_FFT_AlongH:
        return &this->_FFT_V;
        break;

    default:
        return NULL;
        break;
    }
}

template const decx::dsp::fft::_FFT2D_1way_config* 
decx::dsp::fft::_cuda_FFT2D_planner<float>::get_FFT_info(const decx::dsp::fft::FFT_directions) const;
template const decx::dsp::fft::_FFT2D_1way_config*
decx::dsp::fft::_cuda_FFT2D_planner<double>::get_FFT_info(const decx::dsp::fft::FFT_directions) const;


template <typename _type_in>
uint2 decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::get_buffer_dims() const
{
    return this->_buffer_dims;
}

template uint2 decx::dsp::fft::_cuda_FFT2D_planner<float>::get_buffer_dims() const;
template uint2 decx::dsp::fft::_cuda_FFT2D_planner<double>::get_buffer_dims() const;



template <typename _data_type>
void decx::dsp::fft::_cuda_FFT2D_planner<_data_type>::release_buffers(decx::dsp::fft::_cuda_FFT2D_planner<_data_type>* _fake_this)
{
    decx::alloc::_device_dealloc(&_fake_this->_tmp1);
    decx::alloc::_device_dealloc(&_fake_this->_tmp2);
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::release_buffers(decx::dsp::fft::_cuda_FFT2D_planner<float>*);
template void decx::dsp::fft::_cuda_FFT2D_planner<double>::release_buffers(decx::dsp::fft::_cuda_FFT2D_planner<double>*);


template <typename _data_type>
decx::dsp::fft::_cuda_FFT2D_planner<_data_type>::~_cuda_FFT2D_planner()
{
    decx::dsp::fft::_cuda_FFT2D_planner<_data_type>::release_buffers(this);
}

template decx::dsp::fft::_cuda_FFT2D_planner<float>::~_cuda_FFT2D_planner();
template decx::dsp::fft::_cuda_FFT2D_planner<double>::~_cuda_FFT2D_planner();
