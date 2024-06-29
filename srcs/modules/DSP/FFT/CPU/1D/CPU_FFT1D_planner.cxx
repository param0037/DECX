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


#include "CPU_FFT1D_planner.h"


template <typename _data_type>
uint64_t decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_signal_len() const
{
    return this->_signal_length;
}

template uint64_t decx::dsp::fft::cpu_FFT1D_planner<float>::get_signal_len() const;
template uint64_t decx::dsp::fft::cpu_FFT1D_planner<double>::get_signal_len() const;


template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_planner<_data_type>::set_signal_length(const uint64_t signal_length)
{
    this->_signal_length = signal_length;
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::set_signal_length(const uint64_t signal_length);
template void decx::dsp::fft::cpu_FFT1D_planner<double>::set_signal_length(const uint64_t signal_length);



template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_planner<_data_type>::_apart_for_smaller_FFTs(de::DH* handle)
{
    uint64_t _load_equal_target = 0;
    uint32_t _frag_num = 2;

    constexpr uint32_t threshold_fragment = sizeof(_data_type) == 8 ? _MAX_TILING_CPU_FFT_FP64_ : _MAX_TILING_CPU_FFT_FP32_;

    // First find the suitable fragment length and the fragment number
    do {
        _load_equal_target = ceil(pow((double)this->_signal_length, 1.f / (double)_frag_num));
        ++_frag_num;
    } while (_load_equal_target > threshold_fragment);

    std::vector<int> _mask_used;
    for (uint32_t i = 0; i < this->_all_radixes.size(); ++i) {
        _mask_used.push_back(false);
    }

    decx::utils::Fixed_Length_Array<uint32_t> _larger_FFT_lengths;
    _larger_FFT_lengths.define_capacity(this->_all_radixes.size());

    uint32_t larger_FFT_size = 1;
    for (uint32_t i = 0; i < this->_all_radixes.size(); ++i) 
    {
        if (_mask_used[i]) continue;
        uint32_t _current_radix = this->_all_radixes[i];
        
        if (larger_FFT_size * _current_radix > _load_equal_target) {     // If exceed, find another one in the remaining part
            /**
            * The factors are descending in this->_all_radixes. Hence, there is chance to meet the requirement
            * at the end of the queue.
            */
            for (uint32_t _search_dex = i + 1; _search_dex < this->_all_radixes.size(); ++_search_dex)
            {
                if (_mask_used[_search_dex]) continue;
                _current_radix = this->_all_radixes[_search_dex];

                if (larger_FFT_size * _current_radix > _load_equal_target) {     // If still exceed, try the other ones, skip this
                    continue;
                }
                larger_FFT_size *= _current_radix;
                _mask_used[_search_dex] = true;
            }
            _larger_FFT_lengths.emplace_back(larger_FFT_size);
            larger_FFT_size = this->_all_radixes[i];
            continue;
        }
        larger_FFT_size *= _current_radix;
        _mask_used[i] = true;
    }
    if (larger_FFT_size > 1) {
        _larger_FFT_lengths.emplace_back(larger_FFT_size);
    }

    // Adjust
    const uint32_t& _residual = _larger_FFT_lengths[_larger_FFT_lengths.size() - 1];
    for (uint32_t i = 0; i < _larger_FFT_lengths.size() - 1; ++i) {
        if (_larger_FFT_lengths[i] * _residual <= threshold_fragment) {
            _larger_FFT_lengths[i] *= _residual;
            _larger_FFT_lengths.pop_back();
            break;
        }
    }

    this->_smaller_FFTs.define_capacity(_larger_FFT_lengths.size());
    for (uint32_t i = 0; i < _larger_FFT_lengths.size(); ++i) {
        this->_smaller_FFTs.emplace_back(_larger_FFT_lengths[i], handle);
        Check_Runtime_Error(handle);
    }
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::_apart_for_smaller_FFTs(de::DH* handle);
template void decx::dsp::fft::cpu_FFT1D_planner<double>::_apart_for_smaller_FFTs(de::DH* handle);


template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_planner<_data_type>::plan(const uint64_t signal_len, decx::utils::_thr_1D* t1D, de::DH* handle)
{
    constexpr uint32_t alignment = _CPU_FFT_PROC_ALIGN_(_data_type);
    constexpr uint32_t threshold_fragment = sizeof(_data_type) == 8 ? _MAX_TILING_CPU_FFT_FP64_ : _MAX_TILING_CPU_FFT_FP32_;

    this->_signal_length = signal_len;

    this->_permitted_concurrency = decx::cpu::_get_permitted_concurrency();

    this->_without_larger_DFT = decx::dsp::fft::_radix_apart<true>(this->_signal_length, &this->_all_radixes);
    
    this->_smaller_FFTs.define_capacity(this->_all_radixes.size());

    if (this->_signal_length > threshold_fragment) {
        this->_apart_for_smaller_FFTs(handle);
    }
    else {
        this->_smaller_FFTs.emplace_back((uint32_t)this->_signal_length, handle);
    }
    Check_Runtime_Error(handle);

    for (uint32_t i = 0; i < this->_smaller_FFTs.effective_size(); ++i) {
        this->_smaller_FFTs[i].plan(t1D);
    }

    uint64_t _warp_proc_len = 1, _store_pitch = 1;
    for (uint32_t i = 0; i < this->_smaller_FFTs.effective_size(); ++i)
    {
        const uint32_t& _current_radix = this->_smaller_FFTs[i].get_signal_len();
        _warp_proc_len *= _current_radix;
        this->_outer_kernel_info.emplace_back(_current_radix, _warp_proc_len, this->_signal_length, _store_pitch);
        _store_pitch *= _current_radix;
    }
    uint32_t _fraction_num = min(this->_permitted_concurrency, 
        decx::utils::ceil<uint32_t>(this->_signal_length / this->_smaller_FFTs[0].get_signal_len(), alignment));
        
    decx::utils::frag_manager_gen_Nx(this->_smaller_FFTs[0].get_thread_patching_modify(),
        this->_signal_length / this->_smaller_FFTs[0].get_signal_len(),
        _fraction_num, alignment);

    for (uint32_t i = 1; i < this->_smaller_FFTs.effective_size(); ++i) 
    {
        _fraction_num = min(this->_permitted_concurrency, 
            decx::utils::ceil<uint32_t>(this->_outer_kernel_info[i]._store_pitch, alignment));

        decx::utils::frag_manager_gen_Nx(this->_smaller_FFTs[i].get_thread_patching_modify(),
            this->_outer_kernel_info[i]._store_pitch,
            _fraction_num, alignment);
    }

    this->_allocate_spaces(handle);
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::plan(const uint64_t, decx::utils::_thr_1D*, de::DH*);
template void decx::dsp::fft::cpu_FFT1D_planner<double>::plan(const uint64_t, decx::utils::_thr_1D*, de::DH*);


template <typename _data_type>
bool decx::dsp::fft::cpu_FFT1D_planner<_data_type>::changed(const uint64_t signal_len, const uint32_t concurrency) const
{
    return (this->_signal_length ^ signal_len) | 
           (this->_permitted_concurrency ^ concurrency);
}

template bool decx::dsp::fft::cpu_FFT1D_planner<float>::changed(const uint64_t, const uint32_t) const;
template bool decx::dsp::fft::cpu_FFT1D_planner<double>::changed(const uint64_t, const uint32_t) const;



template <typename _data_type>
const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_tile_ptr(const uint32_t _thread_id) const
{
    return &this->_tiles[_thread_id];
}

template const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT1D_planner<float>::get_tile_ptr(const uint32_t _thread_id) const;
template const decx::dsp::fft::FKT1D* decx::dsp::fft::cpu_FFT1D_planner<double>::get_tile_ptr(const uint32_t _thread_id) const;



template <typename _data_type>
void* decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_tmp1_ptr() const
{
    return this->_tmp1.ptr;
}

template void* decx::dsp::fft::cpu_FFT1D_planner<float>::get_tmp1_ptr() const;
template void* decx::dsp::fft::cpu_FFT1D_planner<double>::get_tmp1_ptr() const;


template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_planner<_data_type>::release_buffers(decx::dsp::fft::cpu_FFT1D_planner<_data_type>* _fake_this)
{
    for (uint32_t i = 0; i < _fake_this->_tiles.size(); ++i) {
        _fake_this->_tiles[i].release();
    }

    for (uint32_t i = 0; i < _fake_this->_smaller_FFTs.size(); ++i) {
        _fake_this->_smaller_FFTs[i].~cpu_FFT1D_smaller();
    }

    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_tmp1);
    decx::alloc::_host_virtual_page_dealloc(&_fake_this->_tmp2);
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::release_buffers(decx::dsp::fft::cpu_FFT1D_planner<float>*);
template void decx::dsp::fft::cpu_FFT1D_planner<double>::release_buffers(decx::dsp::fft::cpu_FFT1D_planner<double>*);



template <typename _data_type>
decx::dsp::fft::cpu_FFT1D_planner<_data_type>::~cpu_FFT1D_planner()
{
    decx::dsp::fft::cpu_FFT1D_planner<_data_type>::release_buffers(this);
}

template decx::dsp::fft::cpu_FFT1D_planner<float>::~cpu_FFT1D_planner();
template decx::dsp::fft::cpu_FFT1D_planner<double>::~cpu_FFT1D_planner();



template <typename _data_type>
void* decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_tmp2_ptr() const
{
    return this->_tmp2.ptr;
}

template void* decx::dsp::fft::cpu_FFT1D_planner<float>::get_tmp2_ptr() const;
template void* decx::dsp::fft::cpu_FFT1D_planner<double>::get_tmp2_ptr() const;



template <typename _data_type>
const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_smaller_FFT_info_ptr(const uint32_t _order) const
{
    //return &this->_smaller_FFTs[_order];
    return this->_smaller_FFTs.get_const_ptr(_order);
}

template const decx::dsp::fft::cpu_FFT1D_smaller<float>* decx::dsp::fft::cpu_FFT1D_planner<float>::get_smaller_FFT_info_ptr(const uint32_t) const;
template const decx::dsp::fft::cpu_FFT1D_smaller<double>* decx::dsp::fft::cpu_FFT1D_planner<double>::get_smaller_FFT_info_ptr(const uint32_t) const;


template <typename _data_type>
const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_outer_kernel_info(const uint32_t _order) const
{
    return &this->_outer_kernel_info[_order];
}

template const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_planner<float>::get_outer_kernel_info(const uint32_t) const;
template const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_planner<double>::get_outer_kernel_info(const uint32_t) const;



template <typename _data_type>
uint32_t decx::dsp::fft::cpu_FFT1D_planner<_data_type>::get_kernel_call_num() const
{
    return this->_smaller_FFTs.effective_size();
}

template uint32_t decx::dsp::fft::cpu_FFT1D_planner<float>::get_kernel_call_num() const;
template uint32_t decx::dsp::fft::cpu_FFT1D_planner<double>::get_kernel_call_num() const;



template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_planner<_data_type>::_allocate_spaces(de::DH* handle)
{
    this->_tiles.resize(this->_permitted_concurrency);

    constexpr uint32_t alignment = _CPU_FFT_PROC_ALIGN_(_data_type);
    constexpr uint32_t tile_length = sizeof(_data_type) == 8 ? _MAX_TILING_CPU_FFT_FP64_ : _MAX_TILING_CPU_FFT_FP32_;

    for (uint32_t i = 0; i < this->_permitted_concurrency; ++i) {
        this->_tiles[i].allocate_tile<_data_type>(tile_length, handle);
        Check_Runtime_Error(handle);
    }

    const uint64_t _tmp_alloc_size = decx::utils::align<uint64_t>(this->_signal_length, alignment) * sizeof(_data_type) * 2;
    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp1, _tmp_alloc_size) ||
        decx::alloc::_host_virtual_page_malloc(&this->_tmp2, _tmp_alloc_size)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
}


// smaller FFTs

template <typename _data_type>
decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::cpu_FFT1D_smaller(const uint32_t signal_length, de::DH* handle)
{
    this->_signal_length = signal_length;
    this->_W_table._alloc_table(this->_signal_length, handle);
}

template decx::dsp::fft::cpu_FFT1D_smaller<float>::cpu_FFT1D_smaller(const uint32_t, de::DH*);
template decx::dsp::fft::cpu_FFT1D_smaller<double>::cpu_FFT1D_smaller(const uint32_t, de::DH*);


template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::set_length(const uint32_t signal_length, de::DH* handle)
{
    this->_signal_length = signal_length;
    this->_W_table._alloc_table(this->_signal_length, handle);
}

template void decx::dsp::fft::cpu_FFT1D_smaller<float>::set_length(const uint32_t, de::DH*);
template void decx::dsp::fft::cpu_FFT1D_smaller<double>::set_length(const uint32_t, de::DH*);


template <typename _data_type>
void decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::plan(decx::utils::_thr_1D* t1D)
{
    decx::dsp::fft::_radix_apart<false>(this->_signal_length, &this->_radixes);
    
    uint32_t _store_pitch = 1, _warp_proc_len = 1;
    for (uint32_t i = 0; i < this->_radixes.size(); ++i) {
        const uint32_t& current_radix = this->_radixes[i];
        _warp_proc_len *= current_radix;
        
        this->_kernel_infos.emplace_back(current_radix, _warp_proc_len, this->_signal_length, _store_pitch);
        _store_pitch *= current_radix;
    }
    
    this->_W_table._generate_table(t1D);
}

template void decx::dsp::fft::cpu_FFT1D_smaller<float>::plan(decx::utils::_thr_1D* t1D);
template void decx::dsp::fft::cpu_FFT1D_smaller<double>::plan(decx::utils::_thr_1D* t1D);


template <typename _data_type>
uint32_t decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::get_signal_len() const
{
    return this->_signal_length;
}

template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<float>::get_signal_len() const;
template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<double>::get_signal_len() const;



template <typename _data_type>
uint32_t decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::get_kernel_call_num() const
{
    return this->_radixes.size();
}

template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<float>::get_kernel_call_num() const;
template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<double>::get_kernel_call_num() const;


template <typename _data_type>
const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::get_thread_patching() const
{
    return &this->_thread_dispatch;
}

template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<float>::get_thread_patching() const;
template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<double>::get_thread_patching() const;



template <typename _data_type>
decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::get_thread_patching_modify()
{
    return &this->_thread_dispatch;
}

template decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<float>::get_thread_patching_modify();
template decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<double>::get_thread_patching_modify();


template <typename _data_type>
decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::~cpu_FFT1D_smaller()
{
    this->_W_table._release();
}

template decx::dsp::fft::cpu_FFT1D_smaller<float>::~cpu_FFT1D_smaller();
template decx::dsp::fft::cpu_FFT1D_smaller<double>::~cpu_FFT1D_smaller();



template <typename _data_type>
const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_smaller<_data_type>::get_kernel_info_ptr(const uint32_t _id) const
{
    return &this->_kernel_infos[_id];
}

template const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_smaller<float>::get_kernel_info_ptr(const uint32_t) const;
template const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_smaller<double>::get_kernel_info_ptr(const uint32_t) const;
