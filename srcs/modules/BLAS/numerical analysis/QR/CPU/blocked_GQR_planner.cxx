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

// #define _DECX_CPU_PARTS_
#include "blocked_GQR_planner.h"

static decx::utils::simd::xmm256_reg post_mask256_gen_v8(const uint8_t L_front)
{
    decx::utils::simd::xmm256_reg mask;
    mask._vi = _mm256_setzero_si256();
    for (uint8_t i = L_front; i < 8; ++i) {
        mask._arrui[i] = 0xffffffffU;
    }
    return mask;
}

template <typename _data_type>
void decx::blas::Blocked_GQR_planner<_data_type>::Config(const uint2 block_dims, de::DH* handle)
{
    this->_block_dims = block_dims;
    this->_align_bytes = decx::utils::simd::_get_cpu_simd_align_bytes();
    const uint32_t alignment = this->_align_bytes / sizeof(_data_type);

    // Allocate src_tile
    this->_src_tile._dims.x = decx::utils::align<uint32_t>(block_dims.y, alignment);
    this->_src_tile._dims.y = block_dims.x;
    this->_tile_size = (uint64_t)this->_src_tile._dims.x * (uint64_t)this->_src_tile._dims.y * sizeof(_data_type);
    if (decx::alloc::_host_virtual_page_malloc(&(this->_src_tile._ptr), this->_tile_size)){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
    // Allocate V_tile
    this->_V_tile._dims = this->_src_tile._dims;
    if (decx::alloc::_host_virtual_page_malloc(&(this->_V_tile._ptr), this->_tile_size)){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
    this->_W_tile._dims = this->_src_tile._dims;
    if (decx::alloc::_host_virtual_page_malloc(&(this->_W_tile._ptr), this->_tile_size)){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
    // Allocate array for masks
    if (decx::alloc::_host_virtual_page_malloc(&this->_simd_post_masks, alignment * this->_align_bytes)){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION, ALLOC_FAIL);
        return;
    }
    // Generating masks
    uint8_t* post_mask_ptr = this->_simd_post_masks.GetRawPtr<uint8_t>();
    for (int32_t i = 0; i < alignment; ++i){
        decx::utils::simd::xmm256_reg mask = post_mask256_gen_v8(i);
        _mm256_store_ps((float*)(post_mask_ptr + this->_align_bytes * i), mask._vf);
    }

    // Plan for the transpose config
    this->_tp_ldg_config.config(sizeof(_data_type), 1, 
        this->_block_dims, handle);
}

template void decx::blas::Blocked_GQR_planner<float>::Config(const uint2 block_dims, de::DH* handle);


template <typename _data_type>
void decx::blas::Blocked_GQR_planner<_data_type>::FlushAllTiles()
{
    if (this->_src_tile.IsValid()){
        memset(this->_src_tile.GetRawPtr(), 0, this->_tile_size);
    }
    if (this->_V_tile.IsValid()){
        memset(this->_V_tile.GetRawPtr(), 0, this->_tile_size);
    }
    if (this->_W_tile.IsValid()){
        memset(this->_W_tile.GetRawPtr(), 0, this->_tile_size);
    }
}

template void decx::blas::Blocked_GQR_planner<float>::FlushAllTiles();


static uint32_t calc_proc_len_v(const uint32_t    local_col_id, 
                                const uint8_t     alignment, 
                                const uint32_t    proc_len_v1)
{
    int32_t L_front = local_col_id % (uint32_t)alignment;
    int32_t first_lane = alignment - L_front;
    int32_t is_left = first_lane == 0 ? 0 : 1;
    int32_t post_length = proc_len_v1 - first_lane;
    post_length = post_length < 0 ? 0 : post_length;
    return decx::utils::ceil<uint32_t>(post_length, alignment) + is_left;
}


template <typename _data_type> int32_t 
decx::blas::Blocked_GQR_planner<_data_type>::GetPostMask(const uint32_t L_front, void* p_in) const
{
    if (p_in == nullptr){
        return -1;
    }
    const uint8_t* post_mask_ptr = this->_simd_post_masks.template GetRawPtrConst<uint8_t>();
    switch (this->_align_bytes)
    {
    case 32:
        _mm256_storeu_ps((float*)p_in, _mm256_load_ps((float*)(post_mask_ptr + this->_align_bytes * L_front)));
        break;
    
    default:
        return -1;
        break;
    }
    return 0;
}

template int32_t decx::blas::Blocked_GQR_planner<float>::GetPostMask(const uint32_t L_front, void* p_in) const;


template <typename _data_type>
void decx::blas::Blocked_GQR_planner<_data_type>::Process_HouseHolder()
{
    _data_type* p_src_tile = this->_src_tile.template GetRawPtr<_data_type>();
    _data_type* p_V_tile = this->_V_tile.template GetRawPtr<_data_type>();
    const uint32_t panel_pitch = this->_src_tile._dims.x;

    for (int i = 0; i < this->_block_dims.x; ++i) {
        // Calculate householder reflector
        // printf("i : %d\n", i);
        this->Process_SingleCol_HH(p_src_tile + i * panel_pitch + (i/8), 
                                   p_V_tile + i * panel_pitch + (i/8), 
                                   this->_block_dims.y - i, i);
        if (i < this->_block_dims.x - 1) {
            // Update rest of the panel
            this->ApplyRefactors(p_V_tile + this->_V_tile._dims.x * i + (i/8), 
                                 p_src_tile + this->_src_tile._dims.x * (i+1) + (i/8),
                                 i, make_uint2(_block_dims.x - i - 1, _block_dims.y - i));
        }
    }
}

template void decx::blas::Blocked_GQR_planner<float>::Process_HouseHolder();


template <typename _data_type>
void decx::blas::Blocked_GQR_planner<_data_type>::LoadSrcTile(
        const _data_type* src,
        const uint32_t block_id, 
        const uint32_t pitchsrc_v1,
        decx::utils::_thr_1D* t1D)
{
    this->_tp_ldg_config.transpose_4b_caller(src + block_id * pitchsrc_v1 + block_id * this->_block_dims.x, 
        this->_src_tile.template GetRawPtr<_data_type>(), 
        pitchsrc_v1, 
        this->_src_tile._dims.x, 
        t1D);
}

template void decx::blas::Blocked_GQR_planner<float>::LoadSrcTile(const float* src, const uint32_t block_id, const uint32_t pitchsrc_v1, decx::utils::_thr_1D* t1D);
