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


#include "CPU_FFT2D_planner.h"


template <typename _data_type>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT2D_planner<_data_type>::plan_transpose_configs(de::DH* handle)
{
    this->_transpose_config_1st.config(sizeof(_data_type) * 2, this->_concurrency, this->_signal_dims, handle);
    Check_Runtime_Error(handle);

    this->_transpose_config_2nd.config(sizeof(_type_out), this->_concurrency,
        make_uint2(this->_signal_dims.y, this->_signal_dims.x), handle);
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan_transpose_configs<de::CPf>(de::DH*);
template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan_transpose_configs<uint8_t>(de::DH*);
template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan_transpose_configs<float>(de::DH*);
template void decx::dsp::fft::cpu_FFT2D_planner<double>::plan_transpose_configs<de::CPd>(de::DH*);
template void decx::dsp::fft::cpu_FFT2D_planner<double>::plan_transpose_configs<uint8_t>(de::DH*);
template void decx::dsp::fft::cpu_FFT2D_planner<double>::plan_transpose_configs<double>(de::DH*);


template <typename _data_type>
bool decx::dsp::fft::cpu_FFT2D_planner<_data_type>::changed(const decx::_matrix_layout* src_layout, 
                                                            const decx::_matrix_layout* dst_layout,
                                                            const uint32_t concurrency) const
{
    if (src_layout != NULL && dst_layout != NULL) {
        return (this->_signal_dims.x ^ src_layout->width) |
               (this->_signal_dims.y ^ src_layout->height) |
               (this->_concurrency ^ concurrency) |
               (this->_input_typesize ^ src_layout->_single_element_size) |
               (this->_output_typesize ^ dst_layout->_single_element_size);
    }
    else {
        return 1;
    }
}

template bool decx::dsp::fft::cpu_FFT2D_planner<float>::changed(const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t) const;

template bool decx::dsp::fft::cpu_FFT2D_planner<double>::changed(const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t) const;


// _type_out only valid for IFFT
// For FFT, pass de::CPf when op_mode = fp32; de::CPd when op_mode = fp64.
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

    // If operate in float mode, aligned to 4;
    // If operate in double mode, aligned to 2;
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
    const uint32_t _conc_FFT_1D_H = min(decx::utils::ceil<uint32_t>(this->_signal_dims.y, _alignment), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_thread_dist_FFTH, this->_signal_dims.y, _conc_FFT_1D_H, _alignment);

    // Thread distribution in FFT_W
    const uint32_t _conc_FFT_1D_V = min(decx::utils::ceil<uint32_t>(this->_signal_dims.x, _alignment), this->_concurrency);
    decx::utils::frag_manager_gen_Nx(&this->_thread_dist_FFTV, this->_signal_dims.x, _conc_FFT_1D_V, _alignment);

    const uint32_t _alloc_tiles_num = max(_conc_FFT_1D_H, _conc_FFT_1D_V);
    this->_tiles.define_capacity(_alloc_tiles_num);
    
    const uint32_t _tile_frag_pitch = decx::utils::align<uint32_t>(max(this->_signal_dims.x, this->_signal_dims.y), _alignment);

    for (uint32_t i = 0; i < _alloc_tiles_num; ++i) {
        this->_tiles.emplace_back();
        this->_tiles[i].allocate_tile<_data_type>(_tile_frag_pitch, handle);
        Check_Runtime_Error(handle);
    }

    this->plan_transpose_configs<_type_out>(handle);
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan<de::CPf>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan<float>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<float>::plan<uint8_t>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<double>::plan<de::CPd>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<double>::plan<double>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);

template void decx::dsp::fft::cpu_FFT2D_planner<double>::plan<uint8_t>(const decx::_matrix_layout*,
    const decx::_matrix_layout*, decx::utils::_thread_arrange_1D*, de::DH*);


template <typename _data_type>
void* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_tmp1_ptr() const
{
    return this->_tmp1.ptr;
}

template void* decx::dsp::fft::cpu_FFT2D_planner<float>::get_tmp1_ptr() const;
template void* decx::dsp::fft::cpu_FFT2D_planner<double>::get_tmp1_ptr() const;


template <typename _data_type>
void* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_tmp2_ptr() const
{
    return this->_tmp2.ptr;
}

template void* decx::dsp::fft::cpu_FFT2D_planner<float>::get_tmp2_ptr() const;
template void* decx::dsp::fft::cpu_FFT2D_planner<double>::get_tmp2_ptr() const;

template <typename _data_type>
uint2 decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_signal_dims() const
{
    return this->_signal_dims;
}

template uint2 decx::dsp::fft::cpu_FFT2D_planner<float>::get_signal_dims() const;
template uint2 decx::dsp::fft::cpu_FFT2D_planner<double>::get_signal_dims() const;

template <typename _data_type>
const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_FFTH_info() const
{
    return &this->_FFT_H;
}

template const decx::dsp::fft::cpu_FFT1D_smaller<float>* decx::dsp::fft::cpu_FFT2D_planner<float>::get_FFTH_info() const;
template const decx::dsp::fft::cpu_FFT1D_smaller<double>* decx::dsp::fft::cpu_FFT2D_planner<double>::get_FFTH_info() const;


template <typename _data_type>
const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_FFTV_info() const
{
    return &this->_FFT_V;
}

template const decx::dsp::fft::cpu_FFT1D_smaller<float>* decx::dsp::fft::cpu_FFT2D_planner<float>::get_FFTV_info() const;
template const decx::dsp::fft::cpu_FFT1D_smaller<double>* decx::dsp::fft::cpu_FFT2D_planner<double>::get_FFTV_info() const;


template <typename _data_type>
const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_thread_dist_H() const
{
    return &this->_thread_dist_FFTH;
}

template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<float>::get_thread_dist_H() const;
template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<double>::get_thread_dist_H() const;


template <typename _data_type>
const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_thread_dist_V() const
{
    return &this->_thread_dist_FFTV;
}

template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<float>::get_thread_dist_V() const;
template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT2D_planner<double>::get_thread_dist_V() const;


template <typename _data_type>
const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT2D_planner<_data_type>::get_tile_ptr(const uint32_t _id) const
{
    return this->_tiles.get_const_ptr(_id);
}

template const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT2D_planner<float>::get_tile_ptr(const uint32_t _id) const;
template const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT2D_planner<double>::get_tile_ptr(const uint32_t _id) const;


template <typename _data_type>
void decx::dsp::fft::cpu_FFT2D_planner<_data_type>::release_buffers(decx::dsp::fft::cpu_FFT2D_planner<_data_type>* _fake_this)
{
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_tmp1);
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_tmp2);

    for (uint32_t i = 0; i < _fake_this->_tiles.size(); ++i) {
        _fake_this->_tiles[i].release();
    }
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::release_buffers(decx::dsp::fft::cpu_FFT2D_planner<float>*);
template void decx::dsp::fft::cpu_FFT2D_planner<double>::release_buffers(decx::dsp::fft::cpu_FFT2D_planner<double>*);


template <typename _data_type>
decx::dsp::fft::cpu_FFT2D_planner<_data_type>::~cpu_FFT2D_planner()
{
    decx::dsp::fft::cpu_FFT2D_planner<_data_type>::release_buffers(this);
}

template decx::dsp::fft::cpu_FFT2D_planner<float>::~cpu_FFT2D_planner(); 
template decx::dsp::fft::cpu_FFT2D_planner<double>::~cpu_FFT2D_planner();