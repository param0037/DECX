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

#include "blocked_GQR_planner.h"
#include <Element_wise/common/cpu_element_wise_planner.h>
#define MODULE_TAG "GQR_cpu"


static decx::utils::simd::xmm256_reg post_mask256_gen_v8(const uint8_t L_front)
{
    decx::utils::simd::xmm256_reg mask;
    mask._vi = _mm256_setzero_si256();
    for (uint8_t i = L_front; i < 8; ++i) {
        mask._arrui[i] = 0xffffffffU;
    }
    return mask;
}


static decx::utils::simd::xmm256_reg post_mask256_gen_v4(const uint8_t L_front)
{
    decx::utils::simd::xmm256_reg mask;
    mask._vi = _mm256_setzero_si256();
    for (uint8_t i = L_front; i < 4; ++i) {
        mask._arrull[i] = 0xffffffffffffffffU;
    }
    return mask;
}


template <typename _data_type>
int32_t decx::blas::Blocked_GQR_planner<_data_type>::Config(const uint2 block_dims)
{
    int32_t rval = 0;

    this->_block_dims = block_dims;
    this->_align_bytes = decx::utils::simd::GetCPUSimdAlignBytes();
    const uint32_t alignment = this->_align_bytes / sizeof(_data_type);

    // Allocate src_tile
    this->_src_tile.SetDims(decx::utils::ialign_up<uint32_t>(block_dims.y, alignment),
                            block_dims.x);
    this->_tile_size = (uint64_t)this->_src_tile.GetDims().x * (uint64_t)this->_src_tile.GetDims().y * sizeof(_data_type);
    rval |= this->_src_tile.Allocate(PAGABLE, sizeof(_data_type));

    // Allocate V_tile
    this->_V_tile.SetDims(this->_src_tile.GetDims());
    rval |= this->_V_tile.Allocate(PAGABLE, sizeof(_data_type));

    this->_W_tile.SetDims(this->_src_tile.GetDims());
    rval |= this->_W_tile.Allocate(PAGABLE, sizeof(_data_type));

    // Allocate array for masks
    rval |= this->_simd_post_masks.Allocate(alignment * this->_align_bytes, PAGABLE);

    // Generating masks
    uint8_t* post_mask_ptr = this->_simd_post_masks.GetRawPtr<uint8_t>();
    for (int32_t i = 0; i < alignment; ++i)
    {
        decx::utils::simd::xmm256_reg mask;
        mask._vf = _mm256_setzero_ps();
        switch (alignment)
        {
        case 8:
            mask = post_mask256_gen_v8(i);
            break;
        case 4:
            mask = post_mask256_gen_v4(i);
            break;
        
        default:
            break;
        }
        _mm256_store_ps((float*)(post_mask_ptr + this->_align_bytes * i), mask._vf);
    }

    // Plan for the transpose config
    this->_tp_ldg_config.config(sizeof(_data_type), 16, this->_block_dims);
    this->_tp_ldg_config.TaskMgrRegister(&this->_task_mgr);

    rval |= this->_fmgrs_apply_HH.Allocate((this->_block_dims.x - 1) * sizeof(decx::utils::frag_manager), PAGABLE);
    for (int32_t i = 0; i < block_dims.x - 1; ++i){
        decx::utils::frag_manager_gen(this->_fmgrs_apply_HH + i, this->_block_dims.x - i - 1, DecxGetPermitConcurrency());
    }

    this->_IWY.SetDims(decx::utils::ialign_up<uint32_t>(this->_block_dims.y, alignment), this->_block_dims.y);
    rval |= this->_IWY.Allocate(PAGABLE, sizeof(_data_type));

    decx::utils::frag_manager_gen(&this->_fmgr_updateW, this->_block_dims.y, DecxGetPermitConcurrency());

    rval |= this->Config_W_updator();

    return rval;
}

template int32_t decx::blas::Blocked_GQR_planner<float>::Config(const uint2);
template int32_t decx::blas::Blocked_GQR_planner<double>::Config(const uint2);


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
template void decx::blas::Blocked_GQR_planner<double>::FlushAllTiles();


template <typename _data_type> void
decx::blas::Blocked_GQR_planner<_data_type>::Release()
{
    this->_src_tile.Free();
    this->_V_tile.Free();
    this->_IWY.Free();
    this->_W_tile.Free();
    this->_fmgrs_apply_HH.Free();
    this->_simd_post_masks.Free();
    decx::blas::_cpu_transpose_config::release(&this->_tp_ldg_config);
    
}

template void decx::blas::Blocked_GQR_planner<float>::Release();
template void decx::blas::Blocked_GQR_planner<double>::Release();


template <typename _data_type> int32_t 
decx::blas::Blocked_GQR_planner<_data_type>::GetPostMask(const uint32_t L_front, void* p_in) const
{
    if (p_in == nullptr){
        return -1;
    }
    const uint8_t* post_mask_ptr = this->_simd_post_masks.template GetRawPtrConst<uint8_t>();
    switch (this->_align_bytes)
    {
    case 32:        // AVX256
        _mm256_storeu_ps((float*)p_in, _mm256_load_ps((float*)(post_mask_ptr + this->_align_bytes * L_front)));
        break;
    
    default:
        return -1;
        break;
    }
    return 0;
}

template int32_t decx::blas::Blocked_GQR_planner<float>::GetPostMask(const uint32_t L_front, void* p_in) const;
template int32_t decx::blas::Blocked_GQR_planner<double>::GetPostMask(const uint32_t L_front, void* p_in) const;


template <typename _data_type>
int32_t decx::blas::Blocked_GQR_planner<_data_type>::Config_W_updator()
{
    int32_t rval = 0;
    const uint32_t alignment = this->_align_bytes / sizeof(_data_type); 
    uint32_t plan_nodes_num = decx::utils::idiv_ceil<uint32_t>(this->_block_dims.y - 1, alignment);
    uint32_t aligned_vec_len = decx::utils::ialign_up<uint32_t>(this->_block_dims.y, alignment);

    this->_w_update_helpers.Allocate(plan_nodes_num * sizeof(decx::blas::cpu_MVM_planner<_data_type>), PAGABLE);
    for (int32_t i = 0; i < plan_nodes_num; ++i){
        rval |= this->_w_update_helpers[i].Config(make_uint2(aligned_vec_len - i * alignment, this->_block_dims.y));
        rval |= this->_w_update_helpers[i].TaskMgrSingletonHook(&this->_task_mgr);
    }
    return rval;
}

template int32_t decx::blas::Blocked_GQR_planner<float>::Config_W_updator();
template int32_t decx::blas::Blocked_GQR_planner<double>::Config_W_updator();


template <typename _data_type>
void decx::blas::Blocked_GQR_planner<_data_type>::Process_HouseHolder()
{
    const uint32_t panel_pitch = this->_src_tile.GetDims().x;

    for (int col_id = 0; col_id < this->_block_dims.x; ++col_id) 
    // for (int col_id = 0; col_id < 5; ++col_id) 
    {
        sColHouseHolderTF(this, col_id);

        sApplyReflectors(this, col_id);

        sUpdateW(this, col_id);
    }
}

template void decx::blas::Blocked_GQR_planner<float>::Process_HouseHolder();
template void decx::blas::Blocked_GQR_planner<double>::Process_HouseHolder();


template <typename _data_type>
void decx::blas::Blocked_GQR_planner<_data_type>::LoadSrcTile(
        const _data_type* src,
        const uint32_t block_id, 
        const uint32_t pitchsrc_v1,
        decx::utils::Thr1D* t1D)
{
    if_opt (sizeof(_data_type) == 4) {
        this->_tp_ldg_config.transpose_4b_caller((const float*)(src + block_id * pitchsrc_v1 + block_id * this->_block_dims.x), 
            this->_src_tile.template GetRawPtr<float>(), 
            pitchsrc_v1, 
            this->_src_tile.GetDims().x);
    }
    else if_opt (sizeof(_data_type) == 8) {
        this->_tp_ldg_config.transpose_8b_caller((const double*)(src + block_id * pitchsrc_v1 + block_id * this->_block_dims.x), 
            this->_src_tile.template GetRawPtr<double>(), 
            pitchsrc_v1, 
            this->_src_tile.GetDims().x);
    }
}

template void decx::blas::Blocked_GQR_planner<float>::LoadSrcTile(const float*, const uint32_t, const uint32_t, decx::utils::Thr1D*);
template void decx::blas::Blocked_GQR_planner<double>::LoadSrcTile(const double*, const uint32_t, const uint32_t, decx::utils::Thr1D*);


template <typename _data_type>
_data_type* decx::blas::Blocked_GQR_planner<_data_type>::GetAlignedBufAddr(
    const decx::blas::Blocked_GQR_planner<_data_type>::BlockedGQR_BufType_e buf_type, const uint32_t col_id, const uint32_t row_id)
{
    const uint32_t alignment = this->_align_bytes / sizeof(_data_type);
    switch (buf_type)
    {
    case BlockedGQR_BufType_e::BGQR_Buffer_src:
        return this->_src_tile + decx::utils::ialign_down<uint32_t>(col_id, alignment) + row_id * this->_src_tile.GetDims().x;
    case BlockedGQR_BufType_e::BGQR_Buffer_V:
        return this->_V_tile + decx::utils::ialign_down<uint32_t>(col_id, alignment) + row_id * this->_V_tile.GetDims().x;
    case BlockedGQR_BufType_e::BGQR_Buffer_W:
        return this->_W_tile + decx::utils::ialign_down<uint32_t>(col_id, alignment) + row_id * this->_W_tile.GetDims().x;
    case BlockedGQR_BufType_e::BGQR_Buffer_IWY:
        return this->_IWY + decx::utils::ialign_down<uint32_t>(col_id, alignment) + row_id * this->_IWY.GetDims().x;
    default:
        DECX_LOG_ERR("Invalid buffer type");
        return nullptr;
    }
}

template float* decx::blas::Blocked_GQR_planner<float>::GetAlignedBufAddr(const decx::blas::Blocked_GQR_planner<float>::BlockedGQR_BufType_e, const uint32_t, const uint32_t);
template double* decx::blas::Blocked_GQR_planner<double>::GetAlignedBufAddr(const decx::blas::Blocked_GQR_planner<double>::BlockedGQR_BufType_e, const uint32_t, const uint32_t);
