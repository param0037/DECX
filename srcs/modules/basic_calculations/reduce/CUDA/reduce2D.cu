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


#define _CU_REDUCE2D_MEM_ALIGN_8B_ 2
#define _CU_REDUCE2D_MEM_ALIGN_4B_ 4
#define _CU_REDUCE2D_MEM_ALIGN_2B_ 8
#define _CU_REDUCE2D_MEM_ALIGN_1B_ 16



template <typename _type_in>
uint32_t decx::reduce::cuda_reduce2D_1way_configs<_type_in>::_calc_reduce_kernel_call_times() const
{
    uint16_t _proc_vec_len = 1;
    if (std::is_same<_type_in, float>::value || std::is_same<_type_in, int32_t>::value) {
        _proc_vec_len = _CU_REDUCE2D_MEM_ALIGN_4B_;
    }

    uint32_t _times = 1;
    uint64_t _reduce_len_v = decx::utils::ceil<uint32_t>(this->_proc_dims_actual.x, _proc_vec_len) * _proc_vec_len;

    uint64_t grid_len = decx::utils::ceil<uint64_t>(_reduce_len_v, _REDUCE2D_BLOCK_DIM_X_);

    if (grid_len > 1) {
        uint64_t proc_len_v1 = grid_len;
        _reduce_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
        grid_len = decx::utils::ceil<uint64_t>(_reduce_len_v, _REDUCE2D_BLOCK_DIM_X_);

        while (true) {
            ++_times;
            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            _reduce_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
            grid_len = decx::utils::ceil<uint64_t>(_reduce_len_v, _REDUCE2D_BLOCK_DIM_X_);
        }
    }

    return _times;
}

template uint32_t decx::reduce::cuda_reduce2D_1way_configs<float>::_calc_reduce_kernel_call_times() const;



template <typename _type_in>
template <bool _is_reduce_h>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::generate_configs(const uint2 proc_dims, decx::cuda_stream* S)
{
    this->_proc_dims_actual = proc_dims;

    uint32_t _alloc_dim_x, _grid_len_r1;
    uint16_t _reduce_proc_align;

    if (std::is_same<_type_in, float>::value) {
        _reduce_proc_align = _CU_REDUCE2D_MEM_ALIGN_4B_;
    }

    this->_proc_dims_v = make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, _reduce_proc_align), proc_dims.y);
    _alloc_dim_x = this->_proc_dims_v.x * _reduce_proc_align;

    if (_is_reduce_h) {
        _grid_len_r1 = decx::utils::ceil<uint64_t>(_alloc_dim_x / _reduce_proc_align, _REDUCE2D_BLOCK_DIM_X_);
        this->_d_tmp2._dims = make_uint2(_grid_len_r1, proc_dims.y);

        this->_kernel_call_times = this->_calc_reduce_kernel_call_times();
    }
    else {
        _grid_len_r1 = decx::utils::ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_);
        this->_d_tmp2._dims = make_uint2(_alloc_dim_x, _grid_len_r1);
    }

    this->_d_tmp1._dims = make_uint2(_alloc_dim_x, proc_dims.y);

    if (decx::alloc::_device_malloc(&this->_d_tmp1._ptr, this->_d_tmp1._dims.x * this->_d_tmp1._dims.y * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2._ptr, this->_d_tmp2._dims.x * this->_d_tmp2._dims.y * sizeof(_type_in), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1._ptr.ptr, true);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2._ptr.ptr, false);
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::generate_configs<true>(const uint2, decx::cuda_stream*);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::generate_configs<false>(const uint2, decx::cuda_stream*);


template <typename _Ty>
uint2 decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_actual_proc_dims() const
{
    return this->_proc_dims_actual;
}

template uint2 decx::reduce::cuda_reduce2D_1way_configs<float>::get_actual_proc_dims() const;


template <typename _Ty>
uint2 decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_proc_dims_v() const
{
    return this->_proc_dims_v;
}

template uint2 decx::reduce::cuda_reduce2D_1way_configs<float>::get_proc_dims_v() const;


template <typename _Ty>
decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dtmp1() const
{
    return this->_d_tmp1;
}

template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<float>::get_dtmp1() const;


template <typename _Ty>
decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dtmp2() const
{
    return this->_d_tmp2;
}

template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<float>::get_dtmp2() const;


template <typename _Ty>
void* decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_leading_ptr() const
{
    if (this->_MIF_tmp1.leading) {
        return _MIF_tmp1.mem;
    }
    else {
        return _MIF_tmp2.mem;
    }
}

template void* decx::reduce::cuda_reduce2D_1way_configs<float>::get_leading_ptr() const;

template <typename _Ty>
void* decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_lagging_ptr() const
{
    if (this->_MIF_tmp2.leading) {
        return _MIF_tmp1.mem;
    }
    else {
        return _MIF_tmp2.mem;
    }
}

template void* decx::reduce::cuda_reduce2D_1way_configs<float>::get_lagging_ptr() const;


template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::reverse_MIF_states()
{
    this->_MIF_tmp1.leading = !this->_MIF_tmp1.leading;
    this->_MIF_tmp2.leading = !this->_MIF_tmp2.leading;
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::reverse_MIF_states();

template <typename _Ty>
uint32_t decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_kernel_call_times() const
{
    return _kernel_call_times;
}

template uint32_t decx::reduce::cuda_reduce2D_1way_configs<float>::get_kernel_call_times() const;


template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::release_buffer()
{
    decx::alloc::_device_dealloc(&this->_d_tmp1._ptr);
    decx::alloc::_device_dealloc(&this->_d_tmp2._ptr);
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::release_buffer();