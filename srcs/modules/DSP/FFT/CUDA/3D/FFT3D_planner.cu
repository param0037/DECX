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


#include "FFT3D_planner.cuh"


template <typename _data_type> _CRSR_
void decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::plan(const decx::_tensor_layout* _src_layout, 
                                                         const decx::_tensor_layout* _dst_layout,
                                                         de::DH* handle, decx::cuda_stream* S)
{
    this->_signal_dims.x = _src_layout->depth;
    this->_signal_dims.y = _src_layout->width;
    this->_signal_dims.z = _src_layout->height;

    this->_input_typesize = _src_layout->_single_element_size;
    this->_output_typesize = _dst_layout->_single_element_size;

    // constexpr uint8_t _alignment = 8 / sizeof(_data_type);

    this->_FFT_H.plan(this->_signal_dims.z);
    this->_FFT_D.plan(this->_signal_dims.x);
    this->_FFT_W._1way_FFT_conf.plan(this->_signal_dims.y);
    
    this->_FFT_H._pitchsrc = _src_layout->dp_x_wp;
    this->_FFT_H._pitchtmp = _src_layout->dp_x_wp;
    this->_FFT_H._pitchdst = this->_FFT_H._pitchtmp;

#if _CUDA_FFT3D_restrict_coalesce_
    this->_FFT_W._1way_FFT_conf._pitchsrc = decx::utils::align<uint32_t>(_src_layout->dpitch, 16);
    this->_FFT_W._1way_FFT_conf._pitchdst = decx::utils::align<uint32_t>(_src_layout->dpitch, 16);
    this->_FFT_W._1way_FFT_conf._pitchtmp = decx::utils::align<uint32_t>(_src_layout->dpitch, 16);

    this->_sync_dpitchdst_needed = (this->_FFT_W._1way_FFT_conf._pitchdst != _src_layout->dpitch);
#else
    this->_FFT_W._1way_FFT_conf._pitchsrc = _src_layout->dpitch;
    this->_FFT_W._1way_FFT_conf._pitchdst = _src_layout->dpitch;
    this->_FFT_W._1way_FFT_conf._pitchtmp = _src_layout->dpitch;
#endif

    this->_FFT_W._signal_pitch_src = _src_layout->wpitch;
    this->_FFT_W._signal_pitch_dst = _dst_layout->wpitch;
    this->_FFT_W._parallel = this->_signal_dims.z;

    this->_FFT_D._pitchsrc = _dst_layout->wpitch * _dst_layout->height;
    this->_FFT_D._pitchdst = _dst_layout->wpitch * _dst_layout->height;
    this->_FFT_D._pitchtmp = _dst_layout->wpitch * _dst_layout->height;

    const ulonglong3 _alloc_sizes = make_ulonglong3(this->_FFT_H._pitchtmp * this->_FFT_H.get_signal_len(),
                    this->_FFT_W._1way_FFT_conf._pitchtmp * this->_FFT_W._1way_FFT_conf.get_signal_len() * this->_FFT_W._parallel,
                    this->_FFT_D._pitchtmp * this->_FFT_D.get_signal_len());

    const uint64_t alloc_size = max(max(_alloc_sizes.x, _alloc_sizes.y), _alloc_sizes.z);
    if (decx::alloc::_device_malloc(&this->_tmp1, alloc_size * sizeof(_data_type) * 2, true, S) ||
        decx::alloc::_device_malloc(&this->_tmp2, alloc_size * sizeof(_data_type) * 2, true, S)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
}

template _CRSR_ void decx::dsp::fft::_cuda_FFT3D_planner<float>::plan(const decx::_tensor_layout*,
    const decx::_tensor_layout*, de::DH*, decx::cuda_stream*);

template _CRSR_ void decx::dsp::fft::_cuda_FFT3D_planner<double>::plan(const decx::_tensor_layout*,
    const decx::_tensor_layout*, de::DH*, decx::cuda_stream*);



template <typename _data_type> const decx::dsp::fft::_FFT2D_1way_config* 
decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::get_FFT_info(const decx::dsp::fft::FFT_directions _dir) const
{
    switch (_dir)
    {
    case decx::dsp::fft::FFT_directions::_FFT_AlongH:
        return &this->_FFT_H;
        break;

    case decx::dsp::fft::FFT_directions::_FFT_AlongD:
        return &this->_FFT_D;
        break;

    default:
        return NULL;
        break;
    }
}

template const decx::dsp::fft::_FFT2D_1way_config*
decx::dsp::fft::_cuda_FFT3D_planner<float>::get_FFT_info(const FFT_directions) const;

template const decx::dsp::fft::_FFT2D_1way_config*
decx::dsp::fft::_cuda_FFT3D_planner<double>::get_FFT_info(const FFT_directions) const;


template <typename _data_type> const decx::dsp::fft::_cuda_FFT3D_mid_config* 
decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::get_midFFT_info() const
{
    return &this->_FFT_W;
}

template const decx::dsp::fft::_cuda_FFT3D_mid_config*
decx::dsp::fft::_cuda_FFT3D_planner<float>::get_midFFT_info() const;

template const decx::dsp::fft::_cuda_FFT3D_mid_config*
decx::dsp::fft::_cuda_FFT3D_planner<double>::get_midFFT_info() const;



template <typename _data_type>
bool decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::changed(const decx::_tensor_layout* src_layout, 
                                                            const decx::_tensor_layout* dst_layout) const
{
    return (this->_signal_dims.x ^ src_layout->depth) |
        (this->_signal_dims.y ^ src_layout->width) |
        (this->_signal_dims.z ^ src_layout->height) |
        (this->_input_typesize ^ src_layout->_single_element_size) |
        (this->_output_typesize ^ dst_layout->_single_element_size);
}

template bool decx::dsp::fft::_cuda_FFT3D_planner<float>::changed(const decx::_tensor_layout*, const decx::_tensor_layout*) const;
template bool decx::dsp::fft::_cuda_FFT3D_planner<double>::changed(const decx::_tensor_layout*, const decx::_tensor_layout*) const;


template <typename _data_type>
void decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::release(decx::dsp::fft::_cuda_FFT3D_planner<_data_type>* _fake_this)
{
    decx::alloc::_device_dealloc(&_fake_this->_tmp1);
    decx::alloc::_device_dealloc(&_fake_this->_tmp2);
}

template void decx::dsp::fft::_cuda_FFT3D_planner<float>::release(decx::dsp::fft::_cuda_FFT3D_planner<float>*);
template void decx::dsp::fft::_cuda_FFT3D_planner<double>::release(decx::dsp::fft::_cuda_FFT3D_planner<double>*);


template <typename _data_type>
decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::~_cuda_FFT3D_planner()
{
    decx::dsp::fft::_cuda_FFT3D_planner<_data_type>::release(this);
}

template decx::dsp::fft::_cuda_FFT3D_planner<float>::~_cuda_FFT3D_planner();
template decx::dsp::fft::_cuda_FFT3D_planner<double>::~_cuda_FFT3D_planner();
