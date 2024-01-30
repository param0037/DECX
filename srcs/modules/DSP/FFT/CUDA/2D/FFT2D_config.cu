/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT2D_config.cuh"


template <typename _type_in> _CRSR_
decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::_cuda_FFT2D_planner(const uint2 signal_dims, de::DH* handle)
{
    this->_signal_dims = signal_dims;

    constexpr uint8_t _alignment = 8 / sizeof(_type_in);
    this->_buffer_dims = make_uint2(decx::utils::align<uint32_t>(signal_dims.x, _alignment),
        decx::utils::align<uint32_t>(signal_dims.y, _alignment));

    // Allocate buffers in device
    if (decx::alloc::_device_malloc(&this->_tmp1, this->_buffer_dims.x * this->_buffer_dims.y * sizeof(_type_in) * 2) || 
        decx::alloc::_device_malloc(&this->_tmp2, this->_buffer_dims.x * this->_buffer_dims.y * sizeof(_type_in) * 2)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_ALLOCATION, DEV_ALLOC_FAIL);
        return;
    }
}

template decx::dsp::fft::_cuda_FFT2D_planner<float>::_cuda_FFT2D_planner(const uint2 signal_dims, de::DH* handle);


template <typename _type_in>
void decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::plan(const uint32_t pitchsrc, const uint32_t pitchdst)
{
    this->_FFT_H.plan(this->_signal_dims.x);
    this->_FFT_V.plan(this->_signal_dims.y);

    this->_FFT_V._pitchsrc = pitchsrc;
    this->_FFT_V._pitchtmp = this->_buffer_dims.x;
    this->_FFT_V._pitchdst = this->_buffer_dims.x;

    this->_FFT_H._pitchsrc = this->_buffer_dims.y;
    this->_FFT_H._pitchtmp = this->_buffer_dims.y;
    this->_FFT_H._pitchdst = pitchdst;
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::plan(const uint32_t, const uint32_t);


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



template <typename _type_in> const decx::dsp::fft::_FFT2D_1way_config*
decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::get_FFT_info(const FFT_directions _dir) const
{
    switch (_dir)
    {
    case FFT_directions::_FFT_Horizontal:
        return &this->_FFT_H;
        break;

    case FFT_directions::_FFT_Vertical:
        return &this->_FFT_V;
        break;

    default:
        return NULL;
        break;
    }
}

template const decx::dsp::fft::_FFT2D_1way_config* decx::dsp::fft::_cuda_FFT2D_planner<float>::get_FFT_info(const FFT_directions) const;


template <typename _type_in>
uint2 decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::get_buffer_dims() const
{
    return this->_buffer_dims;
}

template uint2 decx::dsp::fft::_cuda_FFT2D_planner<float>::get_buffer_dims() const;



template <typename _type_in>
void decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::release_buffers()
{
    decx::alloc::_device_dealloc(&this->_tmp1);
    decx::alloc::_device_dealloc(&this->_tmp2);
}

template void decx::dsp::fft::_cuda_FFT2D_planner<float>::release_buffers();


template <typename _type_in>
decx::dsp::fft::_cuda_FFT2D_planner<_type_in>::~_cuda_FFT2D_planner()
{
    this->release_buffers();
}

template decx::dsp::fft::_cuda_FFT2D_planner<float>::~_cuda_FFT2D_planner();
