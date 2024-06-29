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


#include "reduce_callers.cuh"
#include "../../../core/allocators.h"


template <typename _type_in, typename _type_postproc>
bool decx::reduce::reduce2D_flatten_postproc_configs_gen(decx::reduce::cuda_reduce1D_configs<_type_postproc>*   _configs_ptr,
                                                         const uint32_t                                         pirchsrc, 
                                                         const uint2                                            proc_dims_v1, 
                                                         decx::cuda_stream*                                     S)
{
    uint16_t _proc_align = 1;
    if (sizeof(_type_in) == 4) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 2) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 1) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_1B_;
    }
    else if (sizeof(_type_in) == 8) {
        _proc_align = _CU_REDUCE1D_MEM_ALIGN_8B_;
    }

    const dim3 _flatten_K_grid = dim3(decx::utils::ceil<uint32_t>(proc_dims_v1.x, _REDUCE2D_BLOCK_DIM_X_ * _proc_align),
                                      decx::utils::ceil<uint32_t>(proc_dims_v1.y, _REDUCE2D_BLOCK_DIM_Y_));

    uint64_t flatten_len = _flatten_K_grid.x * _flatten_K_grid.y;

    _configs_ptr->generate_configs(flatten_len, S);

    decx::reduce::RWPK_2D& rwpk_flatten = _configs_ptr->get_rwpk_flatten();

    rwpk_flatten._block_dims = dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_);
    rwpk_flatten._grid_dims = _flatten_K_grid;
    rwpk_flatten._calc_pitch_src = pirchsrc / _proc_align;
    rwpk_flatten._calc_proc_dims = proc_dims_v1;

    if (flatten_len == 1) {
        return false;
    }
    else {
        return true;
    }
}

template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<float, float>(decx::reduce::cuda_reduce1D_configs<float>*, const uint32_t, const uint2, decx::cuda_stream*);
template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<de::Half, float>(decx::reduce::cuda_reduce1D_configs<float>*, const uint32_t, const uint2, decx::cuda_stream*);
template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<de::Half, de::Half>(decx::reduce::cuda_reduce1D_configs<de::Half>*, const uint32_t, const uint2, decx::cuda_stream*);
template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<uint8_t, int32_t>(decx::reduce::cuda_reduce1D_configs<int32_t>*, const uint32_t, const uint2, decx::cuda_stream*);
template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<uint8_t, uint8_t>(decx::reduce::cuda_reduce1D_configs<uint8_t>*, const uint32_t, const uint2, decx::cuda_stream*);
template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<int32_t, int32_t>(decx::reduce::cuda_reduce1D_configs<int32_t>*, const uint32_t, const uint2, decx::cuda_stream*);
template bool decx::reduce::reduce2D_flatten_postproc_configs_gen<double, double>(decx::reduce::cuda_reduce1D_configs<double>*, const uint32_t, const uint2, decx::cuda_stream*);


template <typename _type_in>
template <bool _src_from_device>
void decx::reduce::cuda_reduce1D_configs<_type_in>::_calc_kernel_param_packs()
{
    uint16_t _proc_align_tr = 1, _proc_align = 1;
    if (sizeof(_type_in) == 4) {
        _proc_align_tr = _CU_REDUCE1D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 2) {
        _proc_align_tr = _CU_REDUCE1D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 1) {
        _proc_align_tr = _CU_REDUCE1D_MEM_ALIGN_1B_;
    }
    else if (sizeof(_type_in) == 8) {
        _proc_align_tr = _CU_REDUCE1D_MEM_ALIGN_8B_;
    }
    _proc_align = sizeof(_type_in) <= 4 ? _CU_REDUCE1D_MEM_ALIGN_4B_ : _CU_REDUCE1D_MEM_ALIGN_8B_;
    _proc_align = this->_remain_load_byte ? _proc_align_tr : _proc_align;

    uint64_t proc_len_v1 = this->_actual_len;
    uint64_t proc_len_v = decx::utils::ceil<uint32_t>(this->_actual_len, _proc_align_tr);
    uint64_t grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

    const void* read_ptr = NULL;
    void* write_ptr = NULL;

    read_ptr = this->_proc_src;
    if (_src_from_device) {
        write_ptr = this->_d_tmp1.ptr;
    }
    else {
        write_ptr = this->_d_tmp2.ptr;
        this->inverse_mutex_MIF_states();
    }
    
    this->_rwpks.emplace_back(read_ptr,         write_ptr, 
                              grid_len,         _REDUCE1D_BLOCK_DIM_,
                              proc_len_v,       proc_len_v1);

    if (grid_len > 1)
    {
        proc_len_v1 = grid_len;
        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, _proc_align);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = this->get_leading_MIF().mem;
            write_ptr = this->get_lagging_MIF().mem;

            this->_rwpks.emplace_back(read_ptr,         write_ptr, 
                                      grid_len,         _REDUCE1D_BLOCK_DIM_,
                                      proc_len_v,       proc_len_v1);

            this->inverse_mutex_MIF_states();

            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, _proc_align);
            grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);
        }
    }
}


template <typename _type_in>
decx::reduce::cuda_reduce1D_configs<_type_in>::cuda_reduce1D_configs()
{
    this->_remain_load_byte = false;
}

template decx::reduce::cuda_reduce1D_configs<float>::cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<de::Half>::cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<double>::cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<uint8_t>::cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<int32_t>::cuda_reduce1D_configs();


template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::generate_configs(const uint64_t proc_len_v1, decx::cuda_stream* S)
{
    this->_actual_len = proc_len_v1;
    uint64_t _first_grid_len = 0, _aligned_proc_len = 0;

    if (sizeof(_type_in) == 4) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_4B_, _REDUCE1D_BLOCK_DIM_);
    }
    else if (sizeof(_type_in) == 1) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_1B_) * _CU_REDUCE1D_MEM_ALIGN_1B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_1B_, _REDUCE1D_BLOCK_DIM_);
    }
    else if (sizeof(_type_in) == 2) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_2B_) * _CU_REDUCE1D_MEM_ALIGN_2B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_2B_, _REDUCE1D_BLOCK_DIM_);
    }
    else if (sizeof(_type_in) == 8) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_8B_) * _CU_REDUCE1D_MEM_ALIGN_8B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_8B_, _REDUCE1D_BLOCK_DIM_);
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp1, _aligned_proc_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2, _first_grid_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1.ptr, true);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2.ptr, false);

    this->_proc_src = this->_d_tmp1.ptr;
    this->_calc_kernel_param_packs<false>();
    this->_proc_dst = this->get_leading_MIF().mem;
}

template void decx::reduce::cuda_reduce1D_configs<float>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<de::Half>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<double>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<int32_t>::generate_configs(const uint64_t, decx::cuda_stream*);



template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::generate_configs(decx::PtrInfo<void> dev_src, const uint64_t proc_len_v1, decx::cuda_stream* S)
{
    this->_actual_len = proc_len_v1;
    uint64_t _first_grid_len = 0, _aligned_proc_len = 0;

    if (sizeof(_type_in) == 4) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_4B_, _REDUCE1D_BLOCK_DIM_);
    }
    else if (sizeof(_type_in) == 1) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_1B_) * _CU_REDUCE1D_MEM_ALIGN_1B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_1B_, _REDUCE1D_BLOCK_DIM_);
    }
    else if (sizeof(_type_in) == 2) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_2B_) * _CU_REDUCE1D_MEM_ALIGN_2B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_2B_, _REDUCE1D_BLOCK_DIM_);
    }
    else if (sizeof(_type_in) == 8) {
        _aligned_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_8B_) * _CU_REDUCE1D_MEM_ALIGN_8B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(_aligned_proc_len / _CU_REDUCE1D_MEM_ALIGN_8B_, _REDUCE1D_BLOCK_DIM_);
    }

    this->_proc_src = dev_src.ptr;

    if (decx::alloc::_device_malloc(&this->_d_tmp1, _first_grid_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2, _first_grid_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1.ptr, true);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2.ptr, false);

    this->_calc_kernel_param_packs<true>();

    this->_proc_dst = this->get_leading_MIF().mem;
}

template void decx::reduce::cuda_reduce1D_configs<float>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<de::Half>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<double>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<int32_t>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);


template <typename _type_in>
uint64_t decx::reduce::cuda_reduce1D_configs<_type_in>::get_actual_len() const
{
    return this->_actual_len;
}



template <typename _type_in>
_type_in decx::reduce::cuda_reduce1D_configs<_type_in>::get_fill_val() const
{
    return this->_fill_val;
}

template float decx::reduce::cuda_reduce1D_configs<float>::get_fill_val() const;
template int32_t decx::reduce::cuda_reduce1D_configs<int32_t>::get_fill_val() const;
template de::Half decx::reduce::cuda_reduce1D_configs<de::Half>::get_fill_val() const;
template uint8_t decx::reduce::cuda_reduce1D_configs<uint8_t>::get_fill_val() const;
template double decx::reduce::cuda_reduce1D_configs<double>::get_fill_val() const;


template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::inverse_mutex_MIF_states()
{
    this->_MIF_tmp1.leading = !this->_MIF_tmp1.leading;
    this->_MIF_tmp2.leading = !this->_MIF_tmp2.leading;
}


template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::set_fill_val(const _type_in _val)
{
    this->_fill_val = _val;
}

template void decx::reduce::cuda_reduce1D_configs<float>::set_fill_val(const float _val);
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::set_fill_val(const uint8_t _val);
template void decx::reduce::cuda_reduce1D_configs<de::Half>::set_fill_val(const de::Half _val);
template void decx::reduce::cuda_reduce1D_configs<double>::set_fill_val(const double _val);
template void decx::reduce::cuda_reduce1D_configs<int32_t>::set_fill_val(const int32_t _val);



template uint64_t decx::reduce::cuda_reduce1D_configs<float>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<de::Half>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<uint8_t>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<double>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<int32_t>::get_actual_len() const;


template void decx::reduce::cuda_reduce1D_configs<float>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<de::Half>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<double>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<int32_t>::inverse_mutex_MIF_states();


template <typename _type_in>
decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<_type_in>::get_leading_MIF() const
{
    if (this->_MIF_tmp1.leading) {
        return this->_MIF_tmp1;
    }
    else if (this->_MIF_tmp2.leading) {
        return this->_MIF_tmp2;
    }
    else {
        return decx::alloc::MIF<void>(NULL);
    }
}


template <typename _type_in>
decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<_type_in>::get_lagging_MIF() const
{
    if (this->_MIF_tmp1.leading) {
        return this->_MIF_tmp2;
    }
    else if (this->_MIF_tmp2.leading) {
        return this->_MIF_tmp1;
    }
    else {
        return decx::alloc::MIF<void>(NULL);
    }
}

template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<float>::get_leading_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<uint8_t>::get_leading_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<de::Half>::get_leading_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<double>::get_leading_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<int32_t>::get_leading_MIF() const;

template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<float>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<uint8_t>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<de::Half>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<double>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<int32_t>::get_lagging_MIF() const;



template <typename _type_in>
void* decx::reduce::cuda_reduce1D_configs<_type_in>::get_src()
{
    return this->_proc_src;
}

template void* decx::reduce::cuda_reduce1D_configs<float>::get_src();
template void* decx::reduce::cuda_reduce1D_configs<de::Half>::get_src();
template void* decx::reduce::cuda_reduce1D_configs<uint8_t>::get_src();
template void* decx::reduce::cuda_reduce1D_configs<double>::get_src();
template void* decx::reduce::cuda_reduce1D_configs<int32_t>::get_src();


template <typename _type_in>
const void* decx::reduce::cuda_reduce1D_configs<_type_in>::get_dst()
{
    return this->_proc_dst;
}

template const void* decx::reduce::cuda_reduce1D_configs<float>::get_dst();
template const void* decx::reduce::cuda_reduce1D_configs<de::Half>::get_dst();
template const void* decx::reduce::cuda_reduce1D_configs<uint8_t>::get_dst();
template const void* decx::reduce::cuda_reduce1D_configs<double>::get_dst();
template const void* decx::reduce::cuda_reduce1D_configs<int32_t>::get_dst();



template <typename _type_in>
std::vector<decx::reduce::RWPK_1D<_type_in>>& decx::reduce::cuda_reduce1D_configs<_type_in>::get_rwpk()
{
    return this->_rwpks;
}

template std::vector<decx::reduce::RWPK_1D<float>>& decx::reduce::cuda_reduce1D_configs<float>::get_rwpk();
template std::vector<decx::reduce::RWPK_1D<de::Half>>& decx::reduce::cuda_reduce1D_configs<de::Half>::get_rwpk();
template std::vector<decx::reduce::RWPK_1D<uint8_t>>& decx::reduce::cuda_reduce1D_configs<uint8_t>::get_rwpk();
template std::vector<decx::reduce::RWPK_1D<double>>& decx::reduce::cuda_reduce1D_configs<double>::get_rwpk();
template std::vector<decx::reduce::RWPK_1D<int32_t>>& decx::reduce::cuda_reduce1D_configs<int32_t>::get_rwpk();



template <typename _type_in>
decx::reduce::RWPK_2D& decx::reduce::cuda_reduce1D_configs<_type_in>::get_rwpk_flatten()
{
    return this->_rwpk_flatten;
}

template decx::reduce::RWPK_2D& decx::reduce::cuda_reduce1D_configs<float>::get_rwpk_flatten();
template decx::reduce::RWPK_2D& decx::reduce::cuda_reduce1D_configs<de::Half>::get_rwpk_flatten();
template decx::reduce::RWPK_2D& decx::reduce::cuda_reduce1D_configs<uint8_t>::get_rwpk_flatten();
template decx::reduce::RWPK_2D& decx::reduce::cuda_reduce1D_configs<double>::get_rwpk_flatten();
template decx::reduce::RWPK_2D& decx::reduce::cuda_reduce1D_configs<int32_t>::get_rwpk_flatten();




template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::release_buffer()
{
    decx::alloc::_device_dealloc(&this->_d_tmp1);
    decx::alloc::_device_dealloc(&this->_d_tmp2);
}

template void decx::reduce::cuda_reduce1D_configs<float>::release_buffer();
template void decx::reduce::cuda_reduce1D_configs<de::Half>::release_buffer();
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::release_buffer();
template void decx::reduce::cuda_reduce1D_configs<double>::release_buffer();
template void decx::reduce::cuda_reduce1D_configs<int32_t>::release_buffer();



template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::set_fp16_accuracy(const uint32_t _accu_lv)
{
    this->_remain_load_byte = (_accu_lv != decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1);
}

template void decx::reduce::cuda_reduce1D_configs<de::Half>::set_fp16_accuracy(const uint32_t _accu_lv);



template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::set_cmp_or_not(const bool _is_cmp)
{
    this->_remain_load_byte = _is_cmp;
}

template void decx::reduce::cuda_reduce1D_configs<float>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce1D_configs<de::Half>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce1D_configs<double>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce1D_configs<int32_t>::set_cmp_or_not(const bool _is_cmp);




template <typename _type_in>
decx::reduce::cuda_reduce1D_configs<_type_in>::~cuda_reduce1D_configs()
{
    this->release_buffer();
}

template decx::reduce::cuda_reduce1D_configs<float>::~cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<de::Half>::~cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<uint8_t>::~cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<double>::~cuda_reduce1D_configs();
template decx::reduce::cuda_reduce1D_configs<int32_t>::~cuda_reduce1D_configs();