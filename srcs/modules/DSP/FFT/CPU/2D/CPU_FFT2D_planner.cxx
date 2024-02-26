/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "CPU_FFT2D_planner.h"

//
//template <typename _type_in>
//decx::dsp::fft::cpu_FFT2D_planner<_type_in>::cpu_FFT2D_planner(const uint2 signal_dims, de::DH* handle)
//{
//    this->_signal_dims = signal_dims;
//
//    constexpr uint8_t _alignment = std::is_same_v<_type_in, float> ? 4 : 2;
//
//    // Get the smallest allocation size (the minimum size that is able to cover all the alignments)
//    const uint2 _aligned_dims = make_uint2(decx::utils::ceil<uint32_t>(signal_dims.x, _alignment) * _alignment,
//                                           decx::utils::ceil<uint32_t>(signal_dims.y, _alignment) * _alignment);
//
//    const uint64_t _alloc_size = max(_aligned_dims.x * signal_dims.y, signal_dims.x * _aligned_dims.y) * sizeof(_type_in) * 2;
//    
//    //decx::PtrInfo<void> _tmp1, _tmp2;
//    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp1, _alloc_size)) {
//        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
//        return;
//    }
//    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp2, _alloc_size)) {
//        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
//        return;
//    }
//    // Allocate other spaces    
//    this->_FFT_H.set_length(this->_signal_dims.x, handle);
//    Check_Runtime_Error(handle);
//    this->_FFT_V.set_length(this->_signal_dims.y, handle);
//    Check_Runtime_Error(handle);
//}
//
//template decx::dsp::fft::cpu_FFT2D_planner<float>::cpu_FFT2D_planner(const uint2, de::DH*);


template <typename _data_type>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT2D_planner<_data_type>::plan_transpose_configs()
{
    this->_transpose_config_1st = decx::bp::_cpu_transpose_config<8>(this->_signal_dims, this->_concurrency);
    this->_transpose_config_2nd = decx::bp::_cpu_transpose_config<sizeof(_type_out)>(
        make_uint2(this->_signal_dims.y, this->_signal_dims.x), this->_concurrency);
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan_transpose_configs<double>();
template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan_transpose_configs<uint8_t>();
template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan_transpose_configs<float>();


template <typename _data_type>
bool decx::dsp::fft::cpu_FFT2D_planner<_data_type>::changed(const decx::_matrix_layout* src_layout, 
                                                            const decx::_matrix_layout* dst_layout,
                                                            const uint32_t concurrency) const
{
    return (this->_signal_dims.x ^ src_layout->width) |
           (this->_signal_dims.y ^ src_layout->height) |
           (this->_concurrency ^ concurrency) |
           (this->_input_typesize ^ src_layout->_single_element_size) |
           (this->_output_typesize ^ dst_layout->_single_element_size);
}

template bool decx::dsp::fft::cpu_FFT2D_planner<float>::changed(const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t) const;



template <typename _data_type> _CRSR_
template <typename _type_out>
void decx::dsp::fft::cpu_FFT2D_planner<_data_type>::plan(const decx::_matrix_layout* src_layout, 
                                                       const decx::_matrix_layout* dst_layout,
                                                       decx::utils::_thread_arrange_1D* t1D, 
                                                       de::DH* handle)
{
    this->_signal_dims.x = src_layout->width;
    this->_signal_dims.y = src_layout->height;
    this->_concurrency = t1D->total_thread;

    this->_input_typesize = src_layout->_single_element_size;
    this->_output_typesize = dst_layout->_single_element_size;

    constexpr uint8_t _alignment = std::is_same_v<_data_type, float> ? 4 : 2;

    // Get the smallest allocation size (the minimum size that is able to cover all the alignments)
    const uint2 _aligned_dims = make_uint2(decx::utils::ceil<uint32_t>(this->_signal_dims.x, _alignment) * _alignment,
        decx::utils::ceil<uint32_t>(this->_signal_dims.y, _alignment) * _alignment);

    const uint64_t _alloc_size = max(_aligned_dims.x * this->_signal_dims.y, this->_signal_dims.x * _aligned_dims.y)
        * sizeof(_data_type) * 2;

    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp1, _alloc_size)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp2, _alloc_size)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
    // Allocate other spaces
    this->_FFT_H.set_length(this->_signal_dims.x, handle);
    Check_Runtime_Error(handle);
    this->_FFT_V.set_length(this->_signal_dims.y, handle);
    Check_Runtime_Error(handle);


    // Plan for FFTs on two dimensions
    this->_FFT_H.plan(t1D);
    this->_FFT_V.plan(t1D);
    const uint32_t _concurrency = decx::cpu::_get_permitted_concurrency();

    // Thread distribution on FFT_H
    const uint32_t _conc_FFT_1D_H = min(decx::utils::ceil<uint32_t>(this->_signal_dims.y, 4), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_thread_dist_FFTH, this->_signal_dims.y, _conc_FFT_1D_H, 4);

    // Thread distribution in FFT_W
    const uint32_t _conc_FFT_1D_V = min(decx::utils::ceil<uint32_t>(this->_signal_dims.x, 4), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_thread_dist_FFTV, this->_signal_dims.x, _conc_FFT_1D_V, 4);

    const uint32_t _alloc_tiles_num = max(_conc_FFT_1D_H, _conc_FFT_1D_V);
    this->_tiles.define_capacity(_alloc_tiles_num);
    
    const uint32_t _tile_frag_pitch = decx::utils::ceil<uint32_t>(max(this->_signal_dims.x, this->_signal_dims.y), 4) * 4;

    for (uint32_t i = 0; i < _alloc_tiles_num; ++i) {
        this->_tiles.emplace_back();
        this->_tiles[i].allocate_tile(_tile_frag_pitch, handle);
        Check_Runtime_Error(handle);
    }

    this->plan_transpose_configs<_type_out>();
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan<double>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan<float>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan<uint8_t>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);




template <typename _type_in>
void* decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_tmp1_ptr() const
{
    return this->_tmp1.ptr;
}

template void* decx::dsp::fft::cpu_FFT2D_planner<float>::get_tmp1_ptr() const;


template <typename _type_in>
void* decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_tmp2_ptr() const
{
    return this->_tmp2.ptr;
}

template void* decx::dsp::fft::cpu_FFT2D_planner<float>::get_tmp2_ptr() const;



template <typename _type_in>
uint2 decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_signal_dims() const
{
    return this->_signal_dims;
}

template uint2 decx::dsp::fft::cpu_FFT2D_planner<float>::get_signal_dims() const;


template <typename _data_type>
const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_FFTH_info() const
{
    return &this->_FFT_H;
}

template const decx::dsp::fft::cpu_FFT1D_smaller<float>* decx::dsp::fft::cpu_FFT2D_planner<float>::get_FFTH_info() const;


template <typename _type_in>
const decx::dsp::fft::cpu_FFT1D_smaller<_type_in>* decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_FFTV_info() const
{
    return &this->_FFT_V;
}

template const decx::dsp::fft::cpu_FFT1D_smaller<float>* decx::dsp::fft::cpu_FFT2D_planner<float>::get_FFTV_info() const;


template <typename _type_in>
const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_thread_dist_H() const
{
    return &this->_thread_dist_FFTH;
}

template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<float>::get_thread_dist_H() const;


template <typename _type_in>
const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_thread_dist_V() const
{
    return &this->_thread_dist_FFTV;
}

template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<float>::get_thread_dist_V() const;



template <typename _type_in>
const decx::dsp::fft::FKT1D_fp32* decx::dsp::fft::cpu_FFT2D_planner<_type_in>::get_tile_ptr(const uint32_t _id) const
{
    return this->_tiles.get_const_ptr(_id);
}

template const decx::dsp::fft::FKT1D_fp32* decx::dsp::fft::cpu_FFT2D_planner<float>::get_tile_ptr(const uint32_t _id) const;


template <typename _type_in>
void decx::dsp::fft::cpu_FFT2D_planner<_type_in>::release_buffers()
{
    decx::alloc::_host_virtual_page_dealloc(&this->_tmp1);
    decx::alloc::_host_virtual_page_dealloc(&this->_tmp2);

    for (uint32_t i = 0; i < this->_tiles.size(); ++i) {
        this->_tiles[i].release();
    }
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::release_buffers();


template <typename _type_in>
decx::dsp::fft::cpu_FFT2D_planner<_type_in>::~cpu_FFT2D_planner()
{
    this->release_buffers();
}

template decx::dsp::fft::cpu_FFT2D_planner<float>::~cpu_FFT2D_planner(); 
//template decx::dsp::fft::cpu_FFT2D_planner<double>::~cpu_FFT2D_planner();