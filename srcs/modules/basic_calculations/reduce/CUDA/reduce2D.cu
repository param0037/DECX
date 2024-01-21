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
template <bool _src_from_device>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::_calc_kernel_h_param_packs(const bool _remain_load_byte)
{
    decx::reduce::RWPK_2D _rwpk;

    uint16_t _proc_align = 1, _proc_align_tr = 1;
    if (sizeof(_type_in) == 4) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 2) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 1) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_1B_;
    }
    else if (sizeof(_type_in) == 8) {
        _proc_align_tr = _CU_REDUCE1D_MEM_ALIGN_8B_;
    }

    if (this->_remain_load_byte) {
        _proc_align = _proc_align_tr;
    } else {
        _proc_align = sizeof(_type_in) <= 4 ? _CU_REDUCE2D_MEM_ALIGN_4B_ : _CU_REDUCE2D_MEM_ALIGN_8B_;
    }

    _rwpk._src = _src_from_device ? 
                this->get_src()._ptr.ptr : 
                this->get_leading_ptr();

    _rwpk._dst = this->get_lagging_ptr();

    // reverse the buffer states
    this->reverse_MIF_states();

    uint32_t grid_x = decx::utils::ceil<uint32_t>(decx::utils::ceil<uint32_t>(this->get_actual_proc_dims().x, _proc_align_tr), 
                                                  _REDUCE2D_BLOCK_DIM_X_);

    const uint32_t grid_y = decx::utils::ceil<uint32_t>(this->get_actual_proc_dims().y, _REDUCE2D_BLOCK_DIM_Y_);

    uint2 proc_dims_actual = this->get_actual_proc_dims();
    uint32_t Wdsrc_v_varient = _src_from_device ? this->_Wdsrc : this->get_dtmp1()._dims.x;
    Wdsrc_v_varient /= _proc_align_tr;

    uint32_t Wddst_v1_varient = decx::utils::ceil<uint32_t>(grid_x, _proc_align) * _proc_align;
    
    const void* _proc_src_ptr = NULL;

    if (grid_x > 1)
    {
        _rwpk._grid_dims      = dim3(grid_x, grid_y);
        _rwpk._block_dims     = dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_);
        _rwpk._calc_pitch_src = Wdsrc_v_varient;
        _rwpk._calc_pitch_dst = Wddst_v1_varient;
        _rwpk._calc_proc_dims = proc_dims_actual;

        this->_rwpks.push_back(_rwpk);

        proc_dims_actual.x = grid_x;
        Wdsrc_v_varient = decx::utils::ceil<uint32_t>(proc_dims_actual.x, _proc_align);
        grid_x = decx::utils::ceil<uint32_t>(Wdsrc_v_varient, _REDUCE2D_BLOCK_DIM_X_);
        // Align the data to _proc_align for the loading pitch of the next kernel
        Wddst_v1_varient = decx::utils::ceil<uint32_t>(grid_x, _proc_align) * _proc_align;

        // If the grid_dims.x of the next kernel is 1, then exit the loop
        while (grid_x > 1)
        {
            this->_rwpks.emplace_back(dim3(grid_x, grid_y),         dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
                                      this->get_leading_ptr(),      this->get_lagging_ptr(), 
                                      Wdsrc_v_varient,              Wddst_v1_varient, 
                                      proc_dims_actual);

            this->reverse_MIF_states();

            proc_dims_actual.x = grid_x;
            Wdsrc_v_varient = decx::utils::ceil<uint32_t>(proc_dims_actual.x, _proc_align);
            grid_x = decx::utils::ceil<uint32_t>(Wdsrc_v_varient, _REDUCE2D_BLOCK_DIM_X_);
            Wddst_v1_varient = decx::utils::ceil<uint32_t>(grid_x, _proc_align) * _proc_align;
        }

        _proc_src_ptr = this->get_leading_ptr();
    }
    else {
        this->reverse_MIF_states();
        _proc_src_ptr = _rwpk._src;
    }
    
    void* _proc_dst_ptr = _src_from_device ? this->get_dst() : this->get_lagging_ptr();

    /**
    * For the last kernel, there is no future kernel, since the Wddst_v1_varient is not aligned to _proc_align to linearly store
    * the data to the destinated array for linearly copying the data. Hence, no tarnsposing is needed.
    */
    this->_rwpks.emplace_back(dim3(grid_x, grid_y),             dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
                              _proc_src_ptr,                    _proc_dst_ptr, 
                              Wdsrc_v_varient,                  grid_x,
                              proc_dims_actual);
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::_calc_kernel_h_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::_calc_kernel_h_param_packs<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::_calc_kernel_h_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::_calc_kernel_h_param_packs<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::_calc_kernel_h_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::_calc_kernel_h_param_packs<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::_calc_kernel_h_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::_calc_kernel_h_param_packs<false>(const bool);



template <typename _type_in>
template <bool _src_from_device>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::_calc_kernel_v_param_packs(const bool _is_cmp)
{
    uint16_t _proc_align = 1, _proc_align_tr = 1;

    uint2 _proc_dims_v1;

    if (sizeof(_type_in) == 4) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 2) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 8) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_8B_;
    }
    else if (sizeof(_type_in) == 1) {
        _proc_align_tr = _CU_REDUCE2D_MEM_ALIGN_1B_;
    }

    _proc_dims_v1 = this->get_actual_proc_dims();

    if (this->_remain_load_byte) {
        _proc_align = _proc_align_tr;
    } else {
        _proc_align = sizeof(_type_in) <= 4 ? _CU_REDUCE2D_MEM_ALIGN_4B_ : _CU_REDUCE2D_MEM_ALIGN_8B_;
    }

    uint32_t grid_y = decx::utils::ceil<uint32_t>(this->get_actual_proc_dims().y, _REDUCE2D_BLOCK_DIM_Y_);

    // The parameters for the firstly called kernel, especially for the different types (e.g. fp16 -> fp32, uint8 -> int32)
    const uint32_t grid_x_tr = decx::utils::ceil<uint32_t>(this->get_actual_proc_dims().x, _REDUCE2D_BLOCK_DIM_X_ * _proc_align_tr);
    
    const uint32_t Wsrc_v_tr = (_src_from_device ?
                               (this->_Wdsrc) :
                               (this->get_dtmp1()._dims.x)) / _proc_align_tr;
    
    const uint32_t Wdst_v_tr = decx::utils::ceil<uint32_t>(this->get_actual_proc_dims().x, _proc_align);

    /**
    * The parameters for the remaining kernels. Since the datatype remains the same.
    * (_proc_align_tr / _proc_align) -> How many times are the two different alignments of datatypes
    */
    const uint32_t grid_x_st = decx::utils::ceil<uint32_t>(this->get_actual_proc_dims().x, _REDUCE2D_BLOCK_DIM_X_ * _proc_align);
    const uint32_t Wsrc_v_st = Wdst_v_tr;
    const uint32_t Wdst_v_st = Wsrc_v_st;

    decx::reduce::RWPK_2D _rwpk;

    // Records the iterating times
    uint32_t _loop_times = 0;
    while (true)
    {
        if (_src_from_device) {
            _rwpk._src = (_loop_times == 0) ? this->get_src()._ptr.ptr : this->get_leading_ptr();
        }
        else {
            _rwpk._src = this->get_leading_ptr();
        }

        _rwpk._dst = this->get_lagging_ptr();

        _rwpk._grid_dims = dim3((_loop_times == 0) ? grid_x_tr : grid_x_st, grid_y);
        _rwpk._block_dims = dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_);
        _rwpk._calc_pitch_src = (_loop_times == 0) ? Wsrc_v_tr : Wsrc_v_st;
        _rwpk._calc_pitch_dst = (_loop_times == 0) ? Wdst_v_tr : Wdst_v_st;
        _rwpk._calc_proc_dims = _proc_dims_v1;

        this->_rwpks.push_back(_rwpk);

        if (grid_y == 1) {
            break;
        }

        this->reverse_MIF_states();
        _proc_dims_v1.y = grid_y;

        grid_y = decx::utils::ceil<uint32_t>(_proc_dims_v1.y, _REDUCE2D_BLOCK_DIM_Y_);

        ++_loop_times;
    }

    if (_src_from_device) {
        this->_rwpks[this->_rwpks.size() - 1]._dst = this->get_dst();
    }
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::_calc_kernel_v_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::_calc_kernel_v_param_packs<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::_calc_kernel_v_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::_calc_kernel_v_param_packs<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::_calc_kernel_v_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::_calc_kernel_v_param_packs<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::_calc_kernel_v_param_packs<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::_calc_kernel_v_param_packs<false>(const bool);




template <typename _type_in>
template <bool _is_reduce_h>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::generate_configs(const uint2 proc_dims, decx::cuda_stream* S, const bool _remain_load_byte)
{
    this->_proc_dims_actual = proc_dims;

    uint32_t _alloc_dim_x, _grid_len_r1;
    uint16_t _reduce_proc_align;

    if (sizeof(_type_in) == 4) {
        _reduce_proc_align = _CU_REDUCE2D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 2) {
        _reduce_proc_align = _CU_REDUCE2D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 1) {
        _reduce_proc_align = _CU_REDUCE2D_MEM_ALIGN_1B_;
    }

    const uint32_t _reduce_len_s1 = decx::utils::ceil<uint32_t>(proc_dims.x, _reduce_proc_align);
    _alloc_dim_x = _reduce_len_s1 * _reduce_proc_align;

    if (_is_reduce_h) {
        _grid_len_r1 = decx::utils::ceil<uint64_t>(_reduce_len_s1, _REDUCE2D_BLOCK_DIM_X_);
        this->_d_tmp2._dims = make_uint2(_grid_len_r1, proc_dims.y);
    }
    else {
        _grid_len_r1 = decx::utils::ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_);
        this->_d_tmp2._dims = make_uint2(_alloc_dim_x, _grid_len_r1);
    }

    this->_d_tmp1._dims = make_uint2(_alloc_dim_x, proc_dims.y);
    
    uint16_t _alloc_typesize;
    if (_remain_load_byte) {
        _alloc_typesize = sizeof(_type_in);
    }
    else {
        _alloc_typesize = sizeof(_type_in) <= 4 ? sizeof(float) : sizeof(double);
    }
    
    if (decx::alloc::_device_malloc(&this->_d_tmp1._ptr, this->_d_tmp1._dims.x * this->_d_tmp1._dims.y * _alloc_typesize, true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2._ptr, this->_d_tmp2._dims.x * this->_d_tmp2._dims.y * _alloc_typesize, true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1._ptr.ptr, true);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2._ptr.ptr, false);

    this->_proc_src = this->_d_tmp1;

    // calculate the parameters packs for CUDA kernels
    if (_is_reduce_h) {
        this->_calc_kernel_h_param_packs<false>(_remain_load_byte);
    }
    else {
        this->_calc_kernel_v_param_packs<false>(_remain_load_byte);
    }

    this->_proc_dst = this->get_lagging_ptr();
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::generate_configs<true>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::generate_configs<false>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::generate_configs<true>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::generate_configs<false>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::generate_configs<true>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::generate_configs<false>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::generate_configs<true>(const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::generate_configs<false>(const uint2, decx::cuda_stream*, const bool);



template <typename _type_in>
template <bool _is_reduce_h>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::generate_configs(decx::PtrInfo<void> dev_src, void* dst_ptr,
    const uint32_t Wdsrc, const uint2 proc_dims, decx::cuda_stream* S, const bool _remain_load_byte)
{
    this->_proc_dims_actual = proc_dims;

    uint32_t _alloc_dim_x, _grid_len_r1;
    uint16_t _proc_align = 1;

    if (sizeof(_type_in) == 4) {
        _proc_align = _CU_REDUCE2D_MEM_ALIGN_4B_;
    }
    else if (sizeof(_type_in) == 2) {
        _proc_align = _CU_REDUCE2D_MEM_ALIGN_2B_;
    }
    else if (sizeof(_type_in) == 1) {
        _proc_align = _CU_REDUCE2D_MEM_ALIGN_1B_;
    }

    this->_Wdsrc = Wdsrc;

    _alloc_dim_x = decx::utils::ceil<uint32_t>(proc_dims.x, _proc_align) * _proc_align;

    if (_is_reduce_h) {
        _grid_len_r1 = decx::utils::ceil<uint64_t>(_alloc_dim_x / _proc_align, _REDUCE2D_BLOCK_DIM_X_);
        this->_d_tmp2._dims = make_uint2(_grid_len_r1, proc_dims.y);
    }
    else {
        _grid_len_r1 = decx::utils::ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_);
        this->_d_tmp2._dims = make_uint2(_alloc_dim_x, _grid_len_r1);
    }

    uint16_t _alloc_typesize;
    if (this->_remain_load_byte) {
        _alloc_typesize = sizeof(_type_in);
    }
    else {
        _alloc_typesize = sizeof(_type_in) <= 4 ? sizeof(float) : sizeof(double);
    }

    this->_d_tmp1._dims = this->_d_tmp2._dims;

    if (decx::alloc::_device_malloc(&this->_d_tmp1._ptr, this->_d_tmp1._dims.x * this->_d_tmp1._dims.y * _alloc_typesize, true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    if (decx::alloc::_device_malloc(&this->_d_tmp2._ptr, this->_d_tmp2._dims.x * this->_d_tmp2._dims.y * _alloc_typesize, true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }

    this->_MIF_tmp1 = decx::alloc::MIF<void>(this->_d_tmp1._ptr.ptr, false);
    this->_MIF_tmp2 = decx::alloc::MIF<void>(this->_d_tmp2._ptr.ptr, true);

    this->_proc_src._ptr = dev_src;
    this->_proc_dst = dst_ptr;

    // calculate the parameters packs for CUDA kernels
    if (_is_reduce_h) {
        this->_calc_kernel_h_param_packs<true>(_remain_load_byte);
    }
    else {
        this->_calc_kernel_v_param_packs<true>(_remain_load_byte);
    }
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::generate_configs<true>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::generate_configs<false>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::generate_configs<true>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::generate_configs<false>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::generate_configs<true>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::generate_configs<false>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::generate_configs<true>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::generate_configs<false>(decx::PtrInfo<void>, void*, const uint32_t, const uint2, decx::cuda_stream*, const bool);




template <typename _Ty>
uint2 decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_actual_proc_dims() const
{
    return this->_proc_dims_actual;
}

template uint2 decx::reduce::cuda_reduce2D_1way_configs<float>::get_actual_proc_dims() const;
template uint2 decx::reduce::cuda_reduce2D_1way_configs<double>::get_actual_proc_dims() const;
template uint2 decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_actual_proc_dims() const;
template uint2 decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_actual_proc_dims() const;


template <typename _Ty>
decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dtmp1() const
{
    return this->_d_tmp1;
}

template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<float>::get_dtmp1() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<double>::get_dtmp1() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_dtmp1() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_dtmp1() const;


template <typename _Ty>
decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dtmp2() const
{
    return this->_d_tmp2;
}

template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<float>::get_dtmp2() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<double>::get_dtmp2() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_dtmp2() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_dtmp2() const;


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
template void* decx::reduce::cuda_reduce2D_1way_configs<double>::get_leading_ptr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_leading_ptr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_leading_ptr() const;

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
template void* decx::reduce::cuda_reduce2D_1way_configs<double>::get_lagging_ptr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_lagging_ptr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_lagging_ptr() const;


template <typename _Ty>
decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_src() const
{
    return this->_proc_src;
}

template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<float>::get_src() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<double>::get_src() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_src() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_src() const;


template <typename _Ty>
void* decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dst() const
{
    return this->_proc_dst;
}

template void* decx::reduce::cuda_reduce2D_1way_configs<float>::get_dst() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<double>::get_dst() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_dst() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_dst() const;


template <typename _Ty>
const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_rwpks() const
{
    return this->_rwpks;
}

template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<float>::get_rwpks() const;
template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<double>::get_rwpks() const;
template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_rwpks() const;
template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_rwpks() const;



template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::set_cmp_or_not(const bool _is_cmp)
{
    this->_remain_load_byte = _is_cmp;
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::set_cmp_or_not(const bool _is_cmp);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::set_cmp_or_not(const bool _is_cmp);




template <typename _type_in>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::set_fp16_accuracy(const uint32_t _fp16_accu)
{
    this->_remain_load_byte = (_fp16_accu != decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1);
}

template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::set_fp16_accuracy(const uint32_t _fp16_accu);



template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::reverse_MIF_states()
{
    this->_MIF_tmp1.leading = !this->_MIF_tmp1.leading;
    this->_MIF_tmp2.leading = !this->_MIF_tmp2.leading;
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::reverse_MIF_states();
template void decx::reduce::cuda_reduce2D_1way_configs<double>::reverse_MIF_states();
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::reverse_MIF_states();
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::reverse_MIF_states();


template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::release_buffer()
{
    decx::alloc::_device_dealloc(&this->_d_tmp1._ptr);
    decx::alloc::_device_dealloc(&this->_d_tmp2._ptr);

    this->_rwpks.clear();
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::release_buffer();
template void decx::reduce::cuda_reduce2D_1way_configs<double>::release_buffer();
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::release_buffer();
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::release_buffer();


template <typename _Ty>
template <bool _is>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::test()
{
    this->_remain_load_byte = _is;
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::test<true>();
template void decx::reduce::cuda_reduce2D_1way_configs<double>::test<true>();
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::test<true>();
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::test<true>();

template void decx::reduce::cuda_reduce2D_1way_configs<float>::test<false>();
template void decx::reduce::cuda_reduce2D_1way_configs<double>::test<false>();
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::test<false>();
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::test<false>();