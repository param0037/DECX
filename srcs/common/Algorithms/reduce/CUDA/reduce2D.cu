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
#include <allocators.h>


#define _CU_REDUCE2D_MEM_ALIGN_8B_ 2
#define _CU_REDUCE2D_MEM_ALIGN_4B_ 4
#define _CU_REDUCE2D_MEM_ALIGN_2B_ 8
#define _CU_REDUCE2D_MEM_ALIGN_1B_ 16


#define MODULE_TAG "CudaReduce"


template <typename _type_in>
template <bool _src_from_device>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::CalcDPH_KernelParams(const bool _remain_load_byte)
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
    //_proc_align_tr = 128 / sizeof(_type_in);

    if (this->_remain_load_byte) {
        _proc_align = _proc_align_tr;
    } else {
        _proc_align = sizeof(_type_in) <= 4 ? _CU_REDUCE2D_MEM_ALIGN_4B_ : _CU_REDUCE2D_MEM_ALIGN_8B_;
        //_proc_align = sizeof(_type_in) <= 4 ? 32 : 16;
    }

    _rwpk._src = _src_from_device ? 
                (void*)this->GetInputAddr() : 
                this->GetLeadingBufPtr();

    _rwpk._dst = this->GetLaggingBufPtr();

    // reverse the buffer states
    this->_pp_buffer.UpdateStatus();

    uint32_t grid_x = decx::utils::idiv_ceil<uint32_t>(decx::utils::idiv_ceil<uint32_t>(this->get_actual_proc_dims().x, _proc_align_tr), 
                                                  _REDUCE2D_BLOCK_DIM_X_);

    const uint32_t grid_y = decx::utils::idiv_ceil<uint32_t>(this->get_actual_proc_dims().y, _REDUCE2D_BLOCK_DIM_Y_);

    uint2 proc_dims_actual = this->get_actual_proc_dims();
    uint32_t Wdsrc_v_varient = _src_from_device ? this->_Wdsrc : this->get_dtmp1().GetDims().x;
    Wdsrc_v_varient /= _proc_align_tr;

    uint32_t Wddst_v1_varient = decx::utils::idiv_ceil<uint32_t>(grid_x, _proc_align) * _proc_align;
    
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
        Wdsrc_v_varient = decx::utils::idiv_ceil<uint32_t>(proc_dims_actual.x, _proc_align);
        grid_x = decx::utils::idiv_ceil<uint32_t>(Wdsrc_v_varient, _REDUCE2D_BLOCK_DIM_X_);
        // Align the data to _proc_align for the loading pitch of the next kernel
        Wddst_v1_varient = decx::utils::idiv_ceil<uint32_t>(grid_x, _proc_align) * _proc_align;

        // If the grid_dims.x of the next kernel is 1, then exit the loop
        while (grid_x > 1)
        {
            this->_rwpks.emplace_back(dim3(grid_x, grid_y),         dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
                                      this->GetLeadingBufPtr(),      this->GetLaggingBufPtr(), 
                                      Wdsrc_v_varient,              Wddst_v1_varient, 
                                      proc_dims_actual);

            this->_pp_buffer.UpdateStatus();

            proc_dims_actual.x = grid_x;
            Wdsrc_v_varient = decx::utils::idiv_ceil<uint32_t>(proc_dims_actual.x, _proc_align);
            grid_x = decx::utils::idiv_ceil<uint32_t>(Wdsrc_v_varient, _REDUCE2D_BLOCK_DIM_X_);
            Wddst_v1_varient = decx::utils::idiv_ceil<uint32_t>(grid_x, _proc_align) * _proc_align;
        }

        _proc_src_ptr = this->GetLeadingBufPtr();
    }
    else {
        this->_pp_buffer.UpdateStatus();
        _proc_src_ptr = _rwpk._src;
    }
    
    void* _proc_dst_ptr = _src_from_device ? this->GetOutputAddr() : this->GetLaggingBufPtr();

    /**
    * For the last kernel, there is no future kernel, since the Wddst_v1_varient is not aligned to _proc_align to linearly store
    * the data to the destinated array for linearly copying the data. Hence, no tarnsposing is needed.
    */
    this->_rwpks.emplace_back(dim3(grid_x, grid_y),             dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
                              _proc_src_ptr,                    _proc_dst_ptr, 
                              Wdsrc_v_varient,                  grid_x,
                              proc_dims_actual);
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::CalcDPH_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::CalcDPH_KernelParams<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::CalcDPH_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::CalcDPH_KernelParams<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::CalcDPH_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::CalcDPH_KernelParams<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::CalcDPH_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::CalcDPH_KernelParams<false>(const bool);



template <typename _type_in>
template <bool _src_from_device>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::CalcDPV_KernelParams(const bool _is_cmp)
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

    uint32_t grid_y = decx::utils::idiv_ceil<uint32_t>(this->get_actual_proc_dims().y, _REDUCE2D_BLOCK_DIM_Y_);

    // The parameters for the firstly called kernel, especially for the different types (e.g. fp16 -> fp32, uint8 -> int32)
    const uint32_t grid_x_tr = decx::utils::idiv_ceil<uint32_t>(this->get_actual_proc_dims().x, _REDUCE2D_BLOCK_DIM_X_ * _proc_align_tr);
    
    const uint32_t Wsrc_v_tr = (_src_from_device ?
                               (this->_Wdsrc) :
                               (this->get_dtmp1().GetDims().x)) / _proc_align_tr;
    
    const uint32_t Wdst_v_tr = decx::utils::idiv_ceil<uint32_t>(this->get_actual_proc_dims().x, _proc_align);

    /**
    * The parameters for the remaining kernels. Since the datatype remains the same.
    * (_proc_align_tr / _proc_align) -> How many times are the two different alignments of datatypes
    */
    const uint32_t grid_x_st = decx::utils::idiv_ceil<uint32_t>(this->get_actual_proc_dims().x, _REDUCE2D_BLOCK_DIM_X_ * _proc_align);
    const uint32_t Wsrc_v_st = Wdst_v_tr;
    const uint32_t Wdst_v_st = Wsrc_v_st;

    decx::reduce::RWPK_2D _rwpk;

    // Records the iterating times
    uint32_t _loop_times = 0;
    while (true)
    {
        if (_src_from_device) {
            _rwpk._src = (_loop_times == 0) ? (const void*)(this->GetInputAddr()) : this->GetLeadingBufPtr();
        }
        else {
            _rwpk._src = this->GetLeadingBufPtr();
        }

        _rwpk._dst = this->GetLaggingBufPtr();

        _rwpk._grid_dims = dim3((_loop_times == 0) ? grid_x_tr : grid_x_st, grid_y);
        _rwpk._block_dims = dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_);
        _rwpk._calc_pitch_src = (_loop_times == 0) ? Wsrc_v_tr : Wsrc_v_st;
        _rwpk._calc_pitch_dst = (_loop_times == 0) ? Wdst_v_tr : Wdst_v_st;
        _rwpk._calc_proc_dims = _proc_dims_v1;

        this->_rwpks.push_back(_rwpk);

        if (grid_y == 1) {
            break;
        }

        this->_pp_buffer.UpdateStatus();
        _proc_dims_v1.y = grid_y;

        grid_y = decx::utils::idiv_ceil<uint32_t>(_proc_dims_v1.y, _REDUCE2D_BLOCK_DIM_Y_);

        ++_loop_times;
    }

    if (_src_from_device) {
        this->_rwpks[this->_rwpks.size() - 1]._dst = this->GetOutputAddr();
    }
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::CalcDPV_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<float>::CalcDPV_KernelParams<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::CalcDPV_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::CalcDPV_KernelParams<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::CalcDPV_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::CalcDPV_KernelParams<false>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::CalcDPV_KernelParams<true>(const bool);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::CalcDPV_KernelParams<false>(const bool);




template <typename _type_in>
template <bool _is_reduce_h>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::generate_configs(const uint2 proc_dims, decx::cuda_stream* S, const bool _remain_load_byte)
{
    int32_t rval = 0;

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

    const uint32_t _reduce_len_s1 = decx::utils::idiv_ceil<uint32_t>(proc_dims.x, _reduce_proc_align);
    _alloc_dim_x = _reduce_len_s1 * _reduce_proc_align;

    if (_is_reduce_h) {
        _grid_len_r1 = decx::utils::idiv_ceil<uint64_t>(_reduce_len_s1, _REDUCE2D_BLOCK_DIM_X_);
        this->_d_tmp2.SetDims(_grid_len_r1, proc_dims.y);
    }
    else {
        _grid_len_r1 = decx::utils::idiv_ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_);
        this->_d_tmp2.SetDims(_alloc_dim_x, _grid_len_r1);
    }

    this->_d_tmp1.SetDims(_alloc_dim_x, proc_dims.y);
    
    uint16_t _alloc_typesize;
    if (_remain_load_byte) {
        _alloc_typesize = sizeof(_type_in);
    }
    else {
        _alloc_typesize = sizeof(_type_in) <= 4 ? sizeof(float) : sizeof(double);
    }
    
    rval |= this->_d_tmp1.Allocate(CUDA_DEVICE, _alloc_typesize, de::GetLastError(), true, S);
    rval |= this->_d_tmp2.Allocate(CUDA_DEVICE, _alloc_typesize, de::GetLastError(), true, S);

    this->_pp_buffer = decx::utils::double_buffer_manager((void*)this->_d_tmp1, (void*)this->_d_tmp2);
    this->_pp_buffer.ResetBuf1AsLeading();

    this->_proc_src = this->_d_tmp1;

    // calculate the parameters packs for CUDA kernels
    if (_is_reduce_h) {
        this->CalcDPH_KernelParams<false>(_remain_load_byte);
    }
    else {
        this->CalcDPV_KernelParams<false>(_remain_load_byte);
    }

    this->_proc_dst = this->GetLaggingBufPtr();
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

    _alloc_dim_x = decx::utils::idiv_ceil<uint32_t>(proc_dims.x, _proc_align) * _proc_align;

    if (_is_reduce_h) {
        _grid_len_r1 = decx::utils::idiv_ceil<uint64_t>(_alloc_dim_x / _proc_align, _REDUCE2D_BLOCK_DIM_X_);
        this->_d_tmp2.SetDims(_grid_len_r1, proc_dims.y);
    }
    else {
        _grid_len_r1 = decx::utils::idiv_ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_);
        this->_d_tmp2.SetDims(_alloc_dim_x, _grid_len_r1);
    }

    uint16_t _alloc_typesize;
    if (this->_remain_load_byte) {
        _alloc_typesize = sizeof(_type_in);
    }
    else {
        _alloc_typesize = sizeof(_type_in) <= 4 ? sizeof(float) : sizeof(double);
    }

    this->_d_tmp1.SetDims(this->_d_tmp2.GetDims());
    
    int32_t rval = 0;
    rval |= this->_d_tmp1.Allocate(_alloc_typesize, CUDA_DEVICE, de::GetLastError(), true, S);
    rval |= this->_d_tmp2.Allocate(_alloc_typesize, CUDA_DEVICE, de::GetLastError(), true, S);

    this->_pp_buffer = decx::utils::double_buffer_manager((void*)this->_d_tmp1, (void*)this->_d_tmp2);
    this->_pp_buffer.ResetBuf1AsLeading();

    this->_proc_src.SetPtr((void*)dev_src);
    this->_proc_dst = dst_ptr;

    // calculate the parameters packs for CUDA kernels
    if (_is_reduce_h) {
        this->CalcDPH_KernelParams<true>(_remain_load_byte);
    }
    else {
        this->CalcDPV_KernelParams<true>(_remain_load_byte);
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
decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dtmp1()
{
    return this->_d_tmp1;
}

template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<float>::get_dtmp1();
template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<double>::get_dtmp1();
template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_dtmp1();
template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_dtmp1();


template <typename _Ty>
decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<_Ty>::get_dtmp2()
{
    return this->_d_tmp2;
}

template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<float>::get_dtmp2();
template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<double>::get_dtmp2();
template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<de::Half>::get_dtmp2();
template decx::Ptr2D_Info<void>& decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::get_dtmp2();


template <typename _Ty>
void* decx::reduce::cuda_reduce2D_1way_configs<_Ty>::GetLeadingBufPtr()
{
    return this->_pp_buffer.template GetLeadingBufPtr<void>();
}

template void* decx::reduce::cuda_reduce2D_1way_configs<float>::GetLeadingBufPtr();
template void* decx::reduce::cuda_reduce2D_1way_configs<double>::GetLeadingBufPtr();
template void* decx::reduce::cuda_reduce2D_1way_configs<de::Half>::GetLeadingBufPtr();
template void* decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::GetLeadingBufPtr();

template <typename _Ty>
void* decx::reduce::cuda_reduce2D_1way_configs<_Ty>::GetLaggingBufPtr()
{
    return this->_pp_buffer.template GetLaggingBufPtr<void>();
}

template void* decx::reduce::cuda_reduce2D_1way_configs<float>::GetLaggingBufPtr();
template void* decx::reduce::cuda_reduce2D_1way_configs<double>::GetLaggingBufPtr();
template void* decx::reduce::cuda_reduce2D_1way_configs<de::Half>::GetLaggingBufPtr();
template void* decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::GetLaggingBufPtr();


template <typename _Ty>
decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<_Ty>::GetInputAddr() const
{
    return this->_proc_src;
}

template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<float>::GetInputAddr() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<double>::GetInputAddr() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<de::Half>::GetInputAddr() const;
template decx::Ptr2D_Info<void> decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::GetInputAddr() const;


template <typename _Ty>
void* decx::reduce::cuda_reduce2D_1way_configs<_Ty>::GetOutputAddr() const
{
    return this->_proc_dst;
}

template void* decx::reduce::cuda_reduce2D_1way_configs<float>::GetOutputAddr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<double>::GetOutputAddr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<de::Half>::GetOutputAddr() const;
template void* decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::GetOutputAddr() const;


template <typename _Ty>
const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<_Ty>::GetRWPKs() const
{
    return this->_rwpks;
}

template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<float>::GetRWPKs() const;
template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<double>::GetRWPKs() const;
template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<de::Half>::GetRWPKs() const;
template const std::vector<decx::reduce::cu_reduce2D_1way_param_pack>& decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::GetRWPKs() const;



template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::CMP(const bool _is_cmp)
{
    this->_remain_load_byte = _is_cmp;
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::CMP(const bool _is_cmp);
template void decx::reduce::cuda_reduce2D_1way_configs<double>::CMP(const bool _is_cmp);
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::CMP(const bool _is_cmp);
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::CMP(const bool _is_cmp);




template <typename _type_in>
void decx::reduce::cuda_reduce2D_1way_configs<_type_in>::SetFp16Accuracy(const uint32_t _fp16_accu)
{
    this->_remain_load_byte = (_fp16_accu != decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1);
}

template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::SetFp16Accuracy(const uint32_t _fp16_accu);


template <typename _Ty>
void decx::reduce::cuda_reduce2D_1way_configs<_Ty>::ReleaseBuffer()
{
    this->_d_tmp1.Free();
    this->_d_tmp2.Free();

    this->_rwpks.clear();
}

template void decx::reduce::cuda_reduce2D_1way_configs<float>::ReleaseBuffer();
template void decx::reduce::cuda_reduce2D_1way_configs<double>::ReleaseBuffer();
template void decx::reduce::cuda_reduce2D_1way_configs<de::Half>::ReleaseBuffer();
template void decx::reduce::cuda_reduce2D_1way_configs<uint8_t>::ReleaseBuffer();


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