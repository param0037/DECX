/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "reduce_callers.cuh"
#include "../../../core/allocators.h"


#define _CU_REDUCE1D_MEM_ALIGN_8B_ 2
#define _CU_REDUCE1D_MEM_ALIGN_4B_ 4
#define _CU_REDUCE1D_MEM_ALIGN_2B_ 8
#define _CU_REDUCE1D_MEM_ALIGN_1B_ 16


template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::generate_configs(const uint64_t proc_len_v1, decx::cuda_stream* S)
{
    this->_actual_len = proc_len_v1;
    uint64_t _first_grid_len = 0;

    if (std::is_same<_type_in, float>::value || std::is_same<_type_in, int>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 4, _REDUCE1D_BLOCK_DIM_);
    }
    else if (std::is_same<_type_in, uint8_t>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_1B_) * _CU_REDUCE1D_MEM_ALIGN_1B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 16, _REDUCE1D_BLOCK_DIM_);
    }
    else if (std::is_same<_type_in, de::Half>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_2B_) * _CU_REDUCE1D_MEM_ALIGN_2B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 8, _REDUCE1D_BLOCK_DIM_);
    }
    else if (std::is_same<_type_in, double>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_8B_) * _CU_REDUCE1D_MEM_ALIGN_8B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 2, _REDUCE1D_BLOCK_DIM_);
    }
    
    if (decx::alloc::_device_malloc(&this->_d_tmp1, this->_proc_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2, _first_grid_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1.ptr, true);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2.ptr, false);

}

template void decx::reduce::cuda_reduce1D_configs<float>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<de::Half>::generate_configs(const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<double>::generate_configs(const uint64_t, decx::cuda_stream*);



template <typename _type_in>
void decx::reduce::cuda_reduce1D_configs<_type_in>::generate_configs(decx::PtrInfo<void> dev_src, const uint64_t proc_len_v1, decx::cuda_stream* S)
{
    this->_actual_len = proc_len_v1;
    uint64_t _first_grid_len = 0;

    if (std::is_same<_type_in, float>::value || std::is_same<_type_in, int>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 4, _REDUCE1D_BLOCK_DIM_);
    }
    else if (std::is_same<_type_in, uint8_t>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_1B_) * _CU_REDUCE1D_MEM_ALIGN_1B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 16, _REDUCE1D_BLOCK_DIM_);
    }
    else if (std::is_same<_type_in, de::Half>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_2B_) * _CU_REDUCE1D_MEM_ALIGN_2B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 8, _REDUCE1D_BLOCK_DIM_);
    }
    else if (std::is_same<_type_in, double>::value) {
        this->_proc_len = decx::utils::ceil<uint64_t>(proc_len_v1, _CU_REDUCE1D_MEM_ALIGN_8B_) * _CU_REDUCE1D_MEM_ALIGN_8B_;
        _first_grid_len = decx::utils::ceil<uint64_t>(this->_proc_len / 2, _REDUCE1D_BLOCK_DIM_);
    }

    this->_dev_src = dev_src;

    if (decx::alloc::_device_malloc(&this->_d_tmp1, _first_grid_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2, _first_grid_len * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1.ptr, false);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2.ptr, true);

}

template void decx::reduce::cuda_reduce1D_configs<float>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<de::Half>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_configs<double>::generate_configs(decx::PtrInfo<void>, const uint64_t, decx::cuda_stream*);


template <typename _type_in>
uint64_t decx::reduce::cuda_reduce1D_configs<_type_in>::get_proc_len() const
{
    return this->_proc_len;
}


template <typename _type_in>
uint64_t decx::reduce::cuda_reduce1D_configs<_type_in>::get_actual_len() const
{
    return this->_actual_len;
}

template <typename _type_in>
decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<_type_in>::get_dev_src() const
{
    return this->_dev_src;
}

template <typename _type_in>
decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<_type_in>::get_dev_tmp1() const
{
    return this->_d_tmp1;
}

template <typename _type_in>
decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<_type_in>::get_dev_tmp2() const
{
    return this->_d_tmp2;
}


template <typename _type_in>
_type_in decx::reduce::cuda_reduce1D_configs<_type_in>::get_fill_val() const
{
    return this->_fill_val;
}

template float decx::reduce::cuda_reduce1D_configs<float>::get_fill_val() const;
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


template uint64_t decx::reduce::cuda_reduce1D_configs<float>::get_proc_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<de::Half>::get_proc_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<uint8_t>::get_proc_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<double>::get_proc_len() const;

template uint64_t decx::reduce::cuda_reduce1D_configs<float>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<de::Half>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<uint8_t>::get_actual_len() const;
template uint64_t decx::reduce::cuda_reduce1D_configs<double>::get_actual_len() const;

template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<float>::get_dev_src() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<de::Half>::get_dev_src() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<uint8_t>::get_dev_src() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<double>::get_dev_src() const;

template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<float>::get_dev_tmp1() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<de::Half>::get_dev_tmp1() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<uint8_t>::get_dev_tmp1() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<double>::get_dev_tmp1() const;

template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<float>::get_dev_tmp2() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<de::Half>::get_dev_tmp2() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<uint8_t>::get_dev_tmp2() const;
template decx::PtrInfo<void> decx::reduce::cuda_reduce1D_configs<double>::get_dev_tmp2() const;

template void decx::reduce::cuda_reduce1D_configs<float>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<de::Half>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<uint8_t>::inverse_mutex_MIF_states();
template void decx::reduce::cuda_reduce1D_configs<double>::inverse_mutex_MIF_states();


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

template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<float>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<uint8_t>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<de::Half>::get_lagging_MIF() const;
template decx::alloc::MIF<void> decx::reduce::cuda_reduce1D_configs<double>::get_lagging_MIF() const;


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
