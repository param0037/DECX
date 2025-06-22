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

#include "MVM_planner.h"
#include <configs/config.h>
#include <log_console.h>
#include <SIMD/intrinsics_ops.h>
#define MODULE_TAG "GEMV"


template <typename _data_type>
decx::blas::cpu_MVM_planner<_data_type>::cpu_MVM_planner()
{
    this->_mat_dims = make_uint2(0, 0);
    this->_alignment = 0;
    this->_concurrency = 0;
}

template decx::blas::cpu_MVM_planner<float>::cpu_MVM_planner();


template <typename _data_type>
decx::blas::cpu_MVM_planner<_data_type>::~cpu_MVM_planner()
{}

template decx::blas::cpu_MVM_planner<float>::~cpu_MVM_planner();


template <typename _data_type>
int32_t decx::blas::cpu_MVM_planner<_data_type>::Config(const uint2 mat_dims)
{
    int32_t rval = 0;
    this->_mat_dims = mat_dims;
    const uint32_t simd_align_byte = 32;
    this->_alignment = simd_align_byte / sizeof(_data_type);

    this->_concurrency = decx::cpu::_get_permitted_concurrency();

    // Use only half of L1 cache size in case there is not enough space.
    uint64_t L1_cache_data_percore = (decx::cpu::_get_L1_data_cache_size_per_core() / 2) / sizeof(_data_type);
    uint32_t block_h = 0;

    // Concurrency is distributed along HEIGHT only
    // For WIDTH, U should only care about the tiling size
    const uint32_t aligned_width = decx::utils::ialign_up<uint32_t>(this->_mat_dims.x, this->_alignment);

    if (L1_cache_data_percore > this->_mat_dims.x) {
        decx::utils::frag_manager_gen(&this->_fmgr_L, aligned_width, 1);
        const uint32_t max_block_w = max(this->_fmgr_L.GetFragLen(), this->_fmgr_L.GetLastFragLen());
        block_h = decx::utils::idiv_ceil<uint32_t>(L1_cache_data_percore, max_block_w);
        decx::utils::frag_manager_gen_Nx(&this->_fmgr_H, this->_mat_dims.y, this->_concurrency, block_h);
    }
    else{
        const uint32_t block_w = decx::utils::ialign_down<uint32_t>(L1_cache_data_percore, this->_alignment);
        block_h = decx::utils::idiv_ceil<uint32_t>(L1_cache_data_percore, block_w);
        decx::utils::frag_manager_gen_from_fragLen(&this->_fmgr_L, aligned_width, block_w);
        decx::utils::frag_manager_gen(&this->_fmgr_H, this->_mat_dims.y, this->_concurrency);
    }

    rval |= this->_block_confs_H_perthread.Allocate(this->_fmgr_H.frag_num * sizeof(decx::utils::frag_manager), PAGABLE);
    for (int32_t i = 0; i < this->_fmgr_H.frag_num; i++) {
        decx::utils::frag_manager_gen_from_fragLen(&this->_block_confs_H_perthread[i], this->_fmgr_H.GetFragLenById(i), block_h);
    }

    const uint32_t width_leftover = this->_mat_dims.x % this->_alignment;
    this->AlignMaskGen(width_leftover, simd_align_byte);
    return rval;
}

template int32_t decx::blas::cpu_MVM_planner<float>::Config(const uint2 mat_dims);
// template int32_t decx::blas::cpu_MVM_planner<double>::Config(const uint2 mat_dims);


template <typename _data_type>
void decx::blas::cpu_MVM_planner<_data_type>::AlignMaskGen(const uint32_t width_leftover, const uint32_t simd_align_byte)
{
    const uint32_t mask_1_num = width_leftover == 0 ? this->_alignment : width_leftover;
#ifdef __x86_64__
    if (32 == simd_align_byte) {    // YMM256
        decx::utils::simd::xmm256_reg mask;
        mask._vi = _mm256_setzero_si256();
        int32_t mask_idx = 0;
        switch (this->_alignment)
        {
        case 8:     // sizeof(_daat_type) = 32-bit
            for (mask_idx = 0; mask_idx < mask_1_num; mask_idx++) mask._arrui[mask_idx] = 0xFFFFFFFFU;
            break;
        case 4:     // sizeof(_daat_type) = 64-bit
            for (mask_idx = 0; mask_idx < mask_1_num; mask_idx++) mask._arrull[mask_idx] = 0xFFFFFFFFFFFFFFFFU;
            break;
        case 0:     // sizeof(_daat_type) = 128-bit
            mask._vi = _mm256_set1_epi32(0xFFFFFFFF);
            break;
        default:
            break;
        }
        _mm256_storeu_ps((float*)this->_L_align_mask, mask._vf);
    } else if (16 == simd_align_byte) {
        decx::utils::simd::xmm128_reg mask;
        mask._vi = _mm_setzero_si128();
        int32_t mask_idx = 0;
        switch (this->_alignment)
        {
        case 4:     // sizeof(_daat_type) = 32-bit
            for (mask_idx = 0; mask_idx < mask_1_num; mask_idx++) mask._arrui[mask_idx] = 0xFFFFFFFFU;
            break;
        case 2:     // sizeof(_daat_type) = 64-bit
            for (mask_idx = 0; mask_idx < mask_1_num; mask_idx++) mask._arrull[mask_idx] = 0xFFFFFFFFFFFFFFFFU;
            break;
        case 0:     // sizeof(_daat_type) = 128-bit
            mask._vi = _mm_set1_epi32(0xFFFFFFFF);
            break;
        default:
            break;
        }
        _mm_storeu_ps((float*)this->_L_align_mask, mask._vf);
    }
#else

#endif

}

template void decx::blas::cpu_MVM_planner<float>::AlignMaskGen(const uint32_t width_leftover, const uint32_t simd_align_byte);


template <typename _data_type>
int32_t decx::blas::cpu_MVM_planner<_data_type>::Release(cpu_MVM_planner<_data_type>* _fake_this)
{
    int32_t rval = _fake_this->_block_confs_H_perthread.Free();
    return rval;
}

template int32_t decx::blas::cpu_MVM_planner<float>::Release(cpu_MVM_planner<float>*);
// template int32_t decx::blas::cpu_MVM_planner<double>::Release(cpu_MVM_planner<double>*);