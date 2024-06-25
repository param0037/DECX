/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "CPU_FFT3D_planner.h"


template <typename _data_type>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT3D_planner<_data_type>::plan_transpose_configs()
{
    /*new(&this->_transp_config_MC) decx::bp::_cpu_transpose_MK_config<8>(make_uint2(this->_signal_dims.x, this->_signal_dims.y),
        this->_concurrency,
        this->_signal_dims.z,
        this->_FFT_D._pitchdst * this->_signal_dims.y,
        this->_FFT_W._pitchsrc * this->_signal_dims.x);

    new(&this->_transp_config_MC_back) decx::bp::_cpu_transpose_MK_config<8>(make_uint2(this->_signal_dims.y, this->_signal_dims.x),
        this->_concurrency,
        this->_signal_dims.z,
        this->_FFT_W._pitchdst * this->_signal_dims.x,
        this->_FFT_D._pitchdst * this->_signal_dims.y);

    new(&this->_transp_config) decx::bp::_cpu_transpose_config<8>(make_uint2(this->_FFT_D._pitchdst * this->_signal_dims.y, this->_signal_dims.z),
        this->_concurrency);

    new(&this->_transp_config_back) decx::bp::_cpu_transpose_config<sizeof(_type_out)> (make_uint2(this->_signal_dims.z, this->_dst_layout->dp_x_wp),
        this->_concurrency);*/
}

template void decx::dsp::fft::cpu_FFT3D_planner<float>::plan_transpose_configs<double>();
template void decx::dsp::fft::cpu_FFT3D_planner<float>::plan_transpose_configs<float>();
template void decx::dsp::fft::cpu_FFT3D_planner<float>::plan_transpose_configs<uint8_t>();


template <typename _data_type>
template <typename _type_out>
_CRSR_ void decx::dsp::fft::cpu_FFT3D_planner<_data_type>::plan(decx::utils::_thread_arrange_1D* t1D, 
                                                              const decx::_tensor_layout* src_layout, 
                                                              const decx::_tensor_layout* dst_layout, 
                                                              de::DH* handle)
{
    this->_signal_dims.x = src_layout->depth;
    this->_signal_dims.y = src_layout->width;
    this->_signal_dims.z = src_layout->height;
    this->_concurrency = decx::cpu::_get_permitted_concurrency();

    this->_input_typesize = src_layout->_single_element_size;
    this->_output_typesize = dst_layout->_single_element_size;

    this->_FFT_D._FFT_info.set_length(this->_signal_dims.x, handle);
    Check_Runtime_Error(handle);
    this->_FFT_W._FFT_info.set_length(this->_signal_dims.y, handle);
    Check_Runtime_Error(handle);
    this->_FFT_H._FFT_info.set_length(this->_signal_dims.z, handle);
    Check_Runtime_Error(handle);

    this->_src_layout = src_layout;
    this->_dst_layout = dst_layout;

    this->_FFT_D._FFT_info.plan(t1D);
    this->_FFT_W._FFT_info.plan(t1D);
    this->_FFT_H._FFT_info.plan(t1D);

    this->_aligned_proc_dims.x = decx::utils::align<uint32_t>(this->_signal_dims.x, 4);
    this->_aligned_proc_dims.y = decx::utils::align<uint32_t>(this->_signal_dims.y, 4);
    this->_aligned_proc_dims.z = decx::utils::align<uint32_t>(this->_signal_dims.z, dst_layout->_single_element_size == 1 ? 8 : 4);
    
    uint32_t _FFTD_lane_num_effective, _conc_FFTD_num;
    // Thread distribution for FFT along depth dimension
    _FFTD_lane_num_effective = this->_signal_dims.y * this->_signal_dims.z;
    _conc_FFTD_num = min(decx::utils::ceil<uint32_t>(_FFTD_lane_num_effective, 4), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_FFT_D._f_mgr, _FFTD_lane_num_effective, _conc_FFTD_num, 4);
    this->_FFT_D._FFT_zip_info_LDG.set_attributes(src_layout->wpitch, this->_signal_dims.y);
    this->_FFT_D._FFT_zip_info_STG.set_attributes(this->_signal_dims.y, this->_signal_dims.y);
    this->_FFT_D._pitchsrc = src_layout->dpitch;
    this->_FFT_D._pitchdst = this->_aligned_proc_dims.x;
    
    // Thread distribution for FFT along width dimension
    _FFTD_lane_num_effective = this->_signal_dims.x * this->_signal_dims.z;
    _conc_FFTD_num = min(decx::utils::ceil<uint32_t>(_FFTD_lane_num_effective, 4), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_FFT_W._f_mgr, _FFTD_lane_num_effective, _conc_FFTD_num, 4);
    this->_FFT_W._FFT_zip_info_LDG.set_attributes(this->_signal_dims.x, this->_signal_dims.x);
    this->_FFT_W._FFT_zip_info_STG.set_attributes(this->_signal_dims.x, this->_signal_dims.x);
    this->_FFT_W._pitchsrc = this->_aligned_proc_dims.y;
    this->_FFT_W._pitchdst = this->_FFT_W._pitchsrc;

    // Thread distribution for FFT along height dimension
    _FFTD_lane_num_effective = this->_signal_dims.x * this->_signal_dims.y;
    _conc_FFTD_num = min(decx::utils::ceil<uint32_t>(_FFTD_lane_num_effective, 4), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_FFT_H._f_mgr, _FFTD_lane_num_effective, _conc_FFTD_num, 4);
    this->_FFT_H._FFT_zip_info_LDG.set_attributes(this->_FFT_D._pitchdst, this->_signal_dims.x);
    this->_FFT_H._FFT_zip_info_STG.set_attributes(dst_layout->dpitch, this->_signal_dims.x, dst_layout->wpitch, this->_signal_dims.y);
    this->_FFT_H._pitchsrc = this->_aligned_proc_dims.z;
    this->_FFT_H._pitchdst = this->_FFT_H._pitchsrc;

    this->allocate_buffers(handle);

    this->template plan_transpose_configs<_type_out>();
}

template void decx::dsp::fft::cpu_FFT3D_planner<float>::plan<de::CPf>(decx::utils::_thread_arrange_1D*,
    const decx::_tensor_layout*, const decx::_tensor_layout*, de::DH*);

template void decx::dsp::fft::cpu_FFT3D_planner<float>::plan<float>(decx::utils::_thread_arrange_1D*,
    const decx::_tensor_layout*, const decx::_tensor_layout*, de::DH*);

template void decx::dsp::fft::cpu_FFT3D_planner<float>::plan<uint8_t>(decx::utils::_thread_arrange_1D*,
    const decx::_tensor_layout*, const decx::_tensor_layout*, de::DH*);



template <typename _data_type>
void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<_data_type>::allocate_buffers(de::DH* handle) 
{
    const uint32_t _FFTD_2D_reqH = this->_signal_dims.y * this->_signal_dims.z;
    const uint32_t _FFTW_2D_reqH = this->_signal_dims.x * this->_signal_dims.z;
    const uint32_t _FFTH_2D_reqH = this->_FFT_D._pitchdst * this->_signal_dims.y;

    // Allocate the two buffers
    const uint64_t _buffer_size = max(max((uint64_t)this->_aligned_proc_dims.x * (uint64_t)_FFTD_2D_reqH,
                                          (uint64_t)this->_aligned_proc_dims.y * (uint64_t)_FFTW_2D_reqH),
                                          (uint64_t)this->_aligned_proc_dims.z * (uint64_t)_FFTH_2D_reqH);

    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp1, _buffer_size * sizeof(_data_type) * 2) ||
        decx::alloc::_host_virtual_page_malloc(&this->_tmp2, _buffer_size * sizeof(_data_type) * 2))
    {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }

    const uint32_t _concurrency = decx::cpu::_get_permitted_concurrency();
    //const uint32_t _tile_frag_pitch = decx::utils::align<uint32_t>(max(max(this->_signal_dims.x, this->_signal_dims.y), this->_signal_dims.z), 4);
    const uint32_t _tile_frag_pitch = max(max(this->_signal_dims.x, this->_signal_dims.y), this->_signal_dims.z);

    this->_tiles.define_capacity(_concurrency);
    for (uint32_t i = 0; i < _concurrency; ++i) {
        this->_tiles.emplace_back();
        this->_tiles[i].allocate_tile<_data_type>(_tile_frag_pitch, handle);
        Check_Runtime_Error(handle);
    }
}

template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::allocate_buffers(de::DH*);


template <typename _data_type> const decx::dsp::fft::cpu_FFT3D_subproc<_data_type>*
decx::dsp::fft::cpu_FFT3D_planner<_data_type>::get_subproc(const decx::dsp::fft::FFT_directions proc_dir) const
{
    switch (proc_dir)
    {
    case decx::dsp::fft::FFT_directions::_FFT_AlongD:
        return &this->_FFT_D;
        break;
    case decx::dsp::fft::FFT_directions::_FFT_AlongW:
        return &this->_FFT_W;
        break;
    case decx::dsp::fft::FFT_directions::_FFT_AlongH:
        return &this->_FFT_H;
        break;
    default:
        return NULL;
        break;
    }
}

template const decx::dsp::fft::cpu_FFT3D_subproc<float>*
decx::dsp::fft::cpu_FFT3D_planner<float>::get_subproc(const decx::dsp::fft::FFT_directions) const;


template <typename _data_type> const
decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT3D_planner<_data_type>::get_tile_ptr(const uint32_t _id) const
{
    return this->_tiles.get_const_ptr(_id);
}

template const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT3D_planner<float>::get_tile_ptr(const uint32_t _id) const;


template <typename _data_type>
bool decx::dsp::fft::cpu_FFT3D_planner<_data_type>::changed(const decx::_tensor_layout* src_layout, 
                                                            const decx::_tensor_layout* dst_layout,
                                                            const uint32_t concurrency) const
{
    return (this->_signal_dims.x ^ src_layout->depth) |
        (this->_signal_dims.y ^ src_layout->width) |
        (this->_signal_dims.z ^ src_layout->height) |
        (this->_concurrency ^ concurrency) |
        (this->_input_typesize ^ src_layout->_single_element_size) |
        (this->_output_typesize ^ dst_layout->_single_element_size);
}

template bool decx::dsp::fft::cpu_FFT3D_planner<float>::changed(const decx::_tensor_layout*, const decx::_tensor_layout* , const uint32_t) const;


template <typename _data_type>
void* decx::dsp::fft::cpu_FFT3D_planner<_data_type>::get_tmp1_ptr() const
{
    return this->_tmp1.ptr;
}

template void* decx::dsp::fft::cpu_FFT3D_planner<float>::get_tmp1_ptr() const;


template <typename _data_type>
void* decx::dsp::fft::cpu_FFT3D_planner<_data_type>::get_tmp2_ptr() const
{
    return this->_tmp2.ptr;
}

template void* decx::dsp::fft::cpu_FFT3D_planner<float>::get_tmp2_ptr() const;




template <typename _data_type>
void decx::dsp::fft::cpu_FFT3D_planner<_data_type>::release(decx::dsp::fft::cpu_FFT3D_planner<_data_type>* _fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_tmp1);
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_tmp2);

    for (uint32_t i = 0; i < _fake_this->_tiles.size(); ++i) {
        _fake_this->_tiles[i].release();
    }
}

template void decx::dsp::fft::cpu_FFT3D_planner<float>::release(decx::dsp::fft::cpu_FFT3D_planner<float>*);


template <typename _data_type>
decx::dsp::fft::cpu_FFT3D_planner<_data_type>::~cpu_FFT3D_planner()
{
    decx::dsp::fft::cpu_FFT3D_planner<_data_type>::release(this);
}

template decx::dsp::fft::cpu_FFT3D_planner<float>::~cpu_FFT3D_planner();
