/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT3D_planner.cuh"


/**
* 紧凑维度D，中间维度W，分离维度H
* 对于中间维度(W)的FFT策略：
* 1. 如果 D>=128, threadIdx.x给维度D，FFT执行维度Wx分离维度H->threadIdx.y, 
* 2. 如果 D<128, 则多维装置
*/

template <typename _type_in> _CRSR_
decx::dsp::fft::_cuda_FFT3D_planner<_type_in>::_cuda_FFT3D_planner(const uint3 signal_dims)
{
    this->_signal_dims = signal_dims;
}

template _CRSR_ decx::dsp::fft::_cuda_FFT3D_planner<float>::_cuda_FFT3D_planner(const uint3);



template <typename _type_in> _CRSR_
void decx::dsp::fft::_cuda_FFT3D_planner<_type_in>::plan(const decx::_tensor_layout* _src_layout, 
                                                         const decx::_tensor_layout* _dst_layout,
                                                         de::DH* handle, decx::cuda_stream* S)
{
    constexpr uint8_t _alignment = 8 / sizeof(_type_in);
    /*this->_pitch_DWH = make_uint3(decx::utils::align<uint32_t>(this->_signal_dims.y * this->_signal_dims.z,  _alignment),
                                  decx::utils::align<uint32_t>(this->_signal_dims.x, _alignment),
                                  decx::utils::align<uint32_t>(this->_signal_dims.x * this->_signal_dims.y, _alignment));*/


    this->_FFT_H.plan(this->_signal_dims.z);
    this->_FFT_D.plan(this->_signal_dims.x);
    this->_FFT_W._1way_FFT_conf.plan(this->_signal_dims.y);

    this->_FFT_H._pitchsrc = _src_layout->dp_x_wp;
    this->_FFT_H._pitchtmp = _dst_layout->dpitch * _src_layout->wpitch;
    this->_FFT_H._pitchdst = this->_FFT_H._pitchtmp;

    this->_FFT_W._1way_FFT_conf._pitchsrc = _dst_layout->dpitch;
    this->_FFT_W._1way_FFT_conf._pitchdst = _dst_layout->dpitch;
    this->_FFT_W._1way_FFT_conf._pitchtmp = _dst_layout->dpitch;
    this->_FFT_W._signal_pitch_src = _src_layout->wpitch;
    this->_FFT_W._signal_pitch_dst = _dst_layout->wpitch;
    this->_FFT_W._parallel = this->_signal_dims.z;

    /*this->_FFT_W._1way_FFT_conf._pitchsrc = _src_layout->dpitch;
    this->_FFT_W._1way_FFT_conf._pitchdst = _src_layout->dpitch;
    this->_FFT_W._1way_FFT_conf._pitchtmp = _src_layout->dpitch;
    this->_FFT_W._parallel = _src_layout->height;
    this->_FFT_W._signal_pitch_src = _src_layout->wpitch;
    this->_FFT_W._signal_pitch_dst = _dst_layout->wpitch;*/

    this->_FFT_D._pitchsrc = _dst_layout->wpitch * _dst_layout->height;
    this->_FFT_D._pitchdst = _dst_layout->wpitch * _dst_layout->height;
    this->_FFT_D._pitchtmp = _dst_layout->wpitch * _dst_layout->height;

    const ulonglong3 _alloc_sizes = make_ulonglong3(this->_FFT_H._pitchtmp * this->_FFT_H.get_signal_len(),
                    this->_FFT_W._1way_FFT_conf._pitchtmp * this->_FFT_W._1way_FFT_conf.get_signal_len(),
                    this->_FFT_D._pitchtmp * this->_FFT_D.get_signal_len());

    const uint64_t alloc_size = max(max(_alloc_sizes.x, _alloc_sizes.y), _alloc_sizes.z);
    if (decx::alloc::_device_malloc(&this->_tmp1, alloc_size * sizeof(_type_in) * 2, true, S) ||
        decx::alloc::_device_malloc(&this->_tmp2, alloc_size * sizeof(_type_in) * 2, true, S)) {
        decx::err::handle_error_info_modify<true>(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
}

template _CRSR_ void decx::dsp::fft::_cuda_FFT3D_planner<float>::plan(const decx::_tensor_layout*,
    const decx::_tensor_layout*, de::DH*, decx::cuda_stream*);



template <typename _type_in> const decx::dsp::fft::_FFT2D_1way_config* 
decx::dsp::fft::_cuda_FFT3D_planner<_type_in>::get_FFT_info(const FFT_directions _dir) const
{
    switch (_dir)
    {
    case FFT_directions::_FFT_AlongH:
        return &this->_FFT_H;
        break;

    case FFT_directions::_FFT_AlongD:
        return &this->_FFT_D;
        break;

    default:
        return NULL;
        break;
    }
}

template const decx::dsp::fft::_FFT2D_1way_config*
decx::dsp::fft::_cuda_FFT3D_planner<float>::get_FFT_info(const FFT_directions) const;


template <typename _type_in> const decx::dsp::fft::_cuda_FFT3D_mid_config* 
decx::dsp::fft::_cuda_FFT3D_planner<_type_in>::get_midFFT_info() const
{
    return &this->_FFT_W;
}

template const decx::dsp::fft::_cuda_FFT3D_mid_config*
decx::dsp::fft::_cuda_FFT3D_planner<float>::get_midFFT_info() const;


template <typename _type_in>
void decx::dsp::fft::_cuda_FFT3D_planner<_type_in>::release()
{
    decx::alloc::_device_dealloc(&this->_tmp1);
    decx::alloc::_device_dealloc(&this->_tmp2);
}

template void decx::dsp::fft::_cuda_FFT3D_planner<float>::release();


template <typename _type_in>
decx::dsp::fft::_cuda_FFT3D_planner<_type_in>::~_cuda_FFT3D_planner()
{
    this->release();
}

template decx::dsp::fft::_cuda_FFT3D_planner<float>::~_cuda_FFT3D_planner();
