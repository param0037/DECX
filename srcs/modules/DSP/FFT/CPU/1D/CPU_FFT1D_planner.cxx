﻿/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "CPU_FFT1D_planner.h"


//
//template <typename _type_in>
//decx::dsp::fft::cpu_FFT1D_planner<_type_in>::cpu_FFT1D_planner(const uint64_t signal_length)
//{
//    this->_signal_length = signal_length;
//}
//
//template decx::dsp::fft::cpu_FFT1D_planner<float>::cpu_FFT1D_planner(const uint64_t signal_length);
////template decx::dsp::fft::cpu_FFT1D_planner<double>::cpu_FFT1D_planner(const uint64_t signal_length);


template <typename _type_in>
uint64_t decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_signal_len() const
{
    return this->_signal_length;
}

template uint64_t decx::dsp::fft::cpu_FFT1D_planner<float>::get_signal_len() const;
//template uint64_t decx::dsp::fft::cpu_FFT1D_planner<double>::get_signal_len() const;


template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_planner<_type_in>::set_signal_length(const uint64_t signal_length)
{
    this->_signal_length = signal_length;
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::set_signal_length(const uint64_t signal_length);
//template void decx::dsp::fft::cpu_FFT1D_planner<double>::set_signal_length(const uint64_t signal_length);



template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_planner<_type_in>::_apart_for_smaller_FFTs(de::DH* handle)
{
    uint64_t _load_equal_target = 0;
    uint32_t _frag_num = 2;
    // First find the suitable fragment length and the fragment number
    do {
        _load_equal_target = ceil(pow((double)this->_signal_length, 1.f / (double)_frag_num));
        ++_frag_num;
    } while (_load_equal_target > _MAX_TILING_CPU_FFT_);

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
        if (_larger_FFT_lengths[i] * _residual <= _MAX_TILING_CPU_FFT_) {
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
//template void decx::dsp::fft::cpu_FFT1D_planner<double>::_apart_for_smaller_FFTs(de::DH* handle);




template <>
void decx::dsp::fft::cpu_FFT1D_planner<float>::plan(const uint64_t signal_len, decx::utils::_thr_1D* t1D, de::DH* handle)
{
    this->_signal_length = signal_len;

    this->_permitted_concurrency = decx::cpu::_get_permitted_concurrency();

    this->_without_larger_DFT = decx::dsp::fft::_radix_apart<true>(this->_signal_length, &this->_all_radixes);
    
    this->_smaller_FFTs.define_capacity(this->_all_radixes.size());

    if (this->_signal_length > _MAX_TILING_CPU_FFT_) {
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
        decx::utils::ceil<uint32_t>(this->_signal_length / this->_smaller_FFTs[0].get_signal_len(), 4));
        
    decx::utils::frag_manager_gen_Nx(this->_smaller_FFTs[0].get_thread_patching_modify(),
        this->_signal_length / this->_smaller_FFTs[0].get_signal_len(),
        _fraction_num, 4);

    for (uint32_t i = 1; i < this->_smaller_FFTs.effective_size(); ++i) 
    {
        _fraction_num = min(this->_permitted_concurrency, 
            decx::utils::ceil<uint32_t>(this->_outer_kernel_info[i]._store_pitch, 4));

        decx::utils::frag_manager_gen_Nx(this->_smaller_FFTs[i].get_thread_patching_modify(),
            this->_outer_kernel_info[i]._store_pitch,
            _fraction_num, 4);
    }

    this->_allocate_spaces(handle);
}


template <typename _data_type>
bool decx::dsp::fft::cpu_FFT1D_planner<_data_type>::changed(const uint64_t signal_len, const uint32_t concurrency) const
{
    return (this->_signal_length ^ signal_len) | 
           (this->_permitted_concurrency ^ concurrency);
}

template bool decx::dsp::fft::cpu_FFT1D_planner<float>::changed(const uint64_t, const uint32_t) const;



template <typename _type_in>
const decx::dsp::fft::FKT1D_fp32* decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_tile_ptr(const uint32_t _thread_id) const
{
    return &this->_tiles[_thread_id];
}

template const decx::dsp::fft::FKT1D_fp32* decx::dsp::fft::cpu_FFT1D_planner<float>::get_tile_ptr(const uint32_t _thread_id) const;
//template void* decx::dsp::fft::cpu_FFT1D_planner<double>::get_tile_ptr(const uint32_t _thread_id) const;


template <typename _type_in>
void* decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_tmp1_ptr() const
{
    return this->_tmp1.ptr;
}

template void* decx::dsp::fft::cpu_FFT1D_planner<float>::get_tmp1_ptr() const;
//template void* decx::dsp::fft::cpu_FFT1D_planner<double>::get_tmp1_ptr() const;


template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_planner<_type_in>::release_buffers()
{
    for (uint32_t i = 0; i < this->_tiles.size(); ++i) {
        this->_tiles[i].release();
    }

    for (uint32_t i = 0; i < this->_smaller_FFTs.size(); ++i) {
        this->_smaller_FFTs[i].~cpu_FFT1D_smaller();
    }

    //this->_W_table._release();

    decx::alloc::_host_virtual_page_dealloc(&this->_tmp1);
    decx::alloc::_host_virtual_page_dealloc(&this->_tmp2);
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::release_buffers();
//template void decx::dsp::fft::cpu_FFT1D_planner<double>::release_buffers();



template <typename _type_in>
decx::dsp::fft::cpu_FFT1D_planner<_type_in>::~cpu_FFT1D_planner()
{
    this->release_buffers();
}

template decx::dsp::fft::cpu_FFT1D_planner<float>::~cpu_FFT1D_planner();
//template decx::dsp::fft::cpu_FFT1D_planner<double>::~cpu_FFT1D_planner();



template <typename _type_in>
void* decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_tmp2_ptr() const
{
    return this->_tmp2.ptr;
}

template void* decx::dsp::fft::cpu_FFT1D_planner<float>::get_tmp2_ptr() const;
//template void* decx::dsp::fft::cpu_FFT1D_planner<double>::get_tmp2_ptr() const;



template <typename _type_in>
const decx::dsp::fft::cpu_FFT1D_smaller<_type_in>* decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_smaller_FFT_info_ptr(const uint32_t _order) const
{
    //return &this->_smaller_FFTs[_order];
    return this->_smaller_FFTs.get_const_ptr(_order);
}

template const decx::dsp::fft::cpu_FFT1D_smaller<float>* decx::dsp::fft::cpu_FFT1D_planner<float>::get_smaller_FFT_info_ptr(const uint32_t) const;
//template const decx::dsp::fft::cpu_FFT1D_smaller<double>* decx::dsp::fft::cpu_FFT1D_planner<double>::get_smaller_FFT_info_ptr(const uint32_t) const;


template <typename _type_in>
const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_outer_kernel_info(const uint32_t _order) const
{
    return &this->_outer_kernel_info[_order];
}

template const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_planner<float>::get_outer_kernel_info(const uint32_t) const;
//template const decx::dsp::fft::cpu_FFT1D_smaller<double>* decx::dsp::fft::cpu_FFT1D_planner<double>::get_smaller_FFT_info_ptr(const uint32_t) const;



template <typename _type_in>
uint32_t decx::dsp::fft::cpu_FFT1D_planner<_type_in>::get_kernel_call_num() const
{
    return this->_smaller_FFTs.effective_size();
}

template uint32_t decx::dsp::fft::cpu_FFT1D_planner<float>::get_kernel_call_num() const;
//template uint32_t decx::dsp::fft::cpu_FFT1D_planner<double>::get_kernel_call_num() const;



template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_planner<_type_in>::_allocate_spaces(de::DH* handle)
{
    this->_tiles.resize(this->_permitted_concurrency);

    for (uint32_t i = 0; i < _permitted_concurrency; ++i) {
        this->_tiles[i].allocate_tile(_MAX_TILING_CPU_FFT_, handle);
        Check_Runtime_Error(handle);
    }

    //this->_W_table._alloc_table(this->_signal_length, handle);
    Check_Runtime_Error(handle);

    const uint64_t _tmp_alloc_size = decx::utils::ceil<uint64_t>(this->_signal_length, 4) * 4 * sizeof(double);
    if (decx::alloc::_host_virtual_page_malloc(&this->_tmp1, _tmp_alloc_size) ||
        decx::alloc::_host_virtual_page_malloc(&this->_tmp2, _tmp_alloc_size)) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
}


// smaller FFTs

template <typename _type_in>
decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::cpu_FFT1D_smaller(const uint32_t signal_length, de::DH* handle)
{
    this->_signal_length = signal_length;
    this->_W_table._alloc_table(this->_signal_length, handle);
}

template decx::dsp::fft::cpu_FFT1D_smaller<float>::cpu_FFT1D_smaller(const uint32_t signal_length, de::DH* handle);
//template decx::dsp::fft::cpu_FFT1D_smaller<double>::cpu_FFT1D_smaller(const uint32_t signal_length);


template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::set_length(const uint32_t signal_length, de::DH* handle)
{
    this->_signal_length = signal_length;
    this->_W_table._alloc_table(this->_signal_length, handle);
}

template void decx::dsp::fft::cpu_FFT1D_smaller<float>::set_length(const uint32_t signal_length, de::DH* handle);
//template void decx::dsp::fft::cpu_FFT1D_smaller<double>::set_length(const uint32_t signal_length, de::DH* handle);


template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::plan(decx::utils::_thr_1D* t1D)
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
//template void decx::dsp::fft::cpu_FFT1D_smaller<double>::plan(decx::utils::_thr_1D* t1D);


template <typename _type_in>
uint32_t decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::get_signal_len() const
{
    return this->_signal_length;
}

template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<float>::get_signal_len() const;
//template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<double>::get_signal_len() const;



template <typename _type_in>
uint32_t decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::get_kernel_call_num() const
{
    return this->_radixes.size();
}

template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<float>::get_kernel_call_num() const;
//template uint32_t decx::dsp::fft::cpu_FFT1D_smaller<double>::get_signal_len() const;


template <typename _type_in>
const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::get_thread_patching() const
{
    return &this->_thread_dispatch;
}

template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<float>::get_thread_patching() const;
//template const decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<double>::get_thread_patching() const;



template <typename _type_in>
decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::get_thread_patching_modify()
{
    return &this->_thread_dispatch;
}

template decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<float>::get_thread_patching_modify();
//template decx::utils::frag_manager* decx::dsp::fft::cpu_FFT1D_smaller<double>::get_thread_patching();


template <typename _type_in>
decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::~cpu_FFT1D_smaller()
{
    this->_W_table._release();
}

template decx::dsp::fft::cpu_FFT1D_smaller<float>::~cpu_FFT1D_smaller();
//template decx::dsp::fft::cpu_FFT1D_smaller<double>::~cpu_FFT1D_smaller();



template <typename _type_in>
const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_smaller<_type_in>::get_kernel_info_ptr(const uint32_t _id) const
{
    return &this->_kernel_infos[_id];
}

template const decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_smaller<float>::get_kernel_info_ptr(const uint32_t) const;
//template decx::dsp::fft::FKI1D* decx::dsp::fft::cpu_FFT1D_smaller<double>::get_kernel_info_ptr(const uint32_t) const;
