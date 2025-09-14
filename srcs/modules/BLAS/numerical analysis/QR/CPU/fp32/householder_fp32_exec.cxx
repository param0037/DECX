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

#include "../blocked_GQR_planner.h"
#include <SIMD/intrinsics_ops.h>
#define MODULE_TAG ""


namespace decx
{
namespace blas
{
namespace CPUK{
static void 
HouseHolder_SingleCol_v8_fp32(const float* __restrict p_col,
                              float* __restrict       p_V,
                              const uint32_t          proc_len_v8,
                              const uint32_t          local_col_id,
                              const __m256            front_mask)
{
    int32_t rval = 0;
    const uint32_t L_front = local_col_id % 8;
    const float x0 = p_col[L_front];

    // Calculate norm2 ^ 2
    float pow2_sum_post = 0;
    __m256 sum_v8 = _mm256_setzero_ps();
    for (int i = 0; i < proc_len_v8; ++i) {
        __m256 eles_v8 = _mm256_load_ps(p_col + (i << 3));
        if (i == 0) {
            eles_v8 = _mm256_and_ps(eles_v8, front_mask);
        }
        sum_v8 = _mm256_fmadd_ps(eles_v8, eles_v8, sum_v8);
    }
    pow2_sum_post = decx::utils::simd::_mm256_h_sum(sum_v8);

    float x_norm2 = sqrtf(pow2_sum_post);

    float sign = x0 < 0.f ? 1 : -1;
    pow2_sum_post -= x0 * x0;
    float v0 = p_col[L_front] - sign * x_norm2;
    pow2_sum_post += v0 * v0;
    pow2_sum_post = sqrtf(pow2_sum_post);

    const float _1_pow2_sum_post = 1.f / pow2_sum_post;
    for (int i = 0; i < proc_len_v8; ++i) {
        __m256 eles_v8 = _mm256_load_ps(p_col + (i << 3));
        eles_v8 = _mm256_mul_ps(eles_v8, _mm256_set1_ps(_1_pow2_sum_post));
        _mm256_store_ps(p_V + (i << 3), eles_v8);
    }
    p_V[L_front] = v0 / pow2_sum_post;
    __m256 eles_v8 = _mm256_load_ps(p_V);
    _mm256_store_ps(p_V, _mm256_and_ps(front_mask, eles_v8));
}


_THREAD_FUNCTION_ static void
Apply_Reflectors_v8_fp32(const float* __restrict    Vk, 
                         float* __restrict          panel_next, 
                         const uint32_t             local_col_id,
                         const uint2                proc_dims_v8,
                         const uint32_t             mat_pitch,
                         const __m256               front_mask)
{
// #pragma omp parallel for
    for (int i = 0; i < proc_dims_v8.y; ++i) {
        __m256 sum_v8 = _mm256_setzero_ps();
        float* next_panel_col = panel_next + mat_pitch * i;
        for (int k = 0; k < proc_dims_v8.x; ++k) {
            __m256 vk_v8 = _mm256_load_ps(Vk + (k * 8));
            __m256 AR_v8 = _mm256_load_ps(next_panel_col + (k * 8));
            if (k == 0) {
                AR_v8 = _mm256_and_ps(AR_v8, front_mask);
            }
            sum_v8 = _mm256_fmadd_ps(vk_v8, AR_v8, sum_v8);
        }
        float res = decx::utils::simd::_mm256_h_sum(sum_v8);
        res *= -2;
        for (int k = 0; k < proc_dims_v8.x; ++k) {
            __m256 vk_v8 = _mm256_load_ps(Vk + (k * 8));
            __m256 AR_v8 = _mm256_load_ps(next_panel_col + (k * 8));
            if (k == 0) {
                AR_v8 = _mm256_and_ps(AR_v8, front_mask);
            }
            AR_v8 = _mm256_fmadd_ps(_mm256_set1_ps(res), vk_v8, AR_v8);
            _mm256_store_ps(next_panel_col + (k * 8), AR_v8);
        }
    }
}

}
}
}


template <>
void decx::blas::Blocked_GQR_planner<float>::
sColHouseHolderTF(decx::blas::Blocked_GQR_planner<float>* _fake_this, 
                  const uint32_t                          local_col_id)
{
    const uint32_t proc_len_v1 = _fake_this->_block_dims.y - local_col_id;
    const uint32_t proc_len_v8 = decx::blas::Blocked_GQR_planner<float>::CalcProcLenV(local_col_id, 8, proc_len_v1);
    if (proc_len_v1 == _fake_this->_block_dims.x && local_col_id == _fake_this->_block_dims.x - 1){
        return;
    }
    __m256 mask = _mm256_setzero_ps();
    const uint32_t L_front = local_col_id % 8;
    _fake_this->GetPostMask(L_front, (void*)(&mask));

    decx::blas::CPUK::HouseHolder_SingleCol_v8_fp32(_fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_src, local_col_id, local_col_id),
        _fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_V, local_col_id, local_col_id),
        proc_len_v8, local_col_id, mask);
}


template <> void decx::blas::Blocked_GQR_planner<float>::
sApplyReflectors(decx::blas::Blocked_GQR_planner<float>* fake_this, 
                 const uint32_t local_col_id)
{
    int32_t rval = 0;
    const uint32_t L_front = local_col_id % 8;
    const uint32_t vec_len_v1 = fake_this->_block_dims.y - local_col_id;
    const uint32_t vec_len_v8 = CalcProcLenV(local_col_id, 8, vec_len_v1);
    const uint32_t pitchsrc = fake_this->_src_tile.GetDims().x;

    __m256 mask = _mm256_setzero_ps();
    rval |= fake_this->GetPostMask(L_front, (void*)(&mask));

    if (local_col_id < fake_this->_block_dims.x - 1) 
    {
        const decx::utils::frag_manager* fmgr = fake_this->_fmgrs_apply_HH + local_col_id;
        fake_this->_task_mgr.SetMaxThreadNum(fmgr->GetFragNum());
        fake_this->_task_mgr.SetDispatchMethod(decx::core::ThreadDispatchMethod_e::Dispatch_ByID);
        
        const float* pV = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_V, local_col_id, local_col_id);
        float* pPanel = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_src, local_col_id, local_col_id + 1);

        decx::cpu_ElementWise1D_planner::
        sCaller(decx::blas::CPUK::Apply_Reflectors_v8_fp32, fmgr, &fake_this->_task_mgr, 
            EW_SLOT_ID_MONOTONIC(0),
            decx::TArg_still<const float*>(pV),
            decx::TArg_var<float*>      ([&](const int32_t i){return pPanel + i * fmgr->GetFragLenById(0) * pitchsrc;}),
            decx::TArg_still<int32_t>(local_col_id),
            decx::TArg_var<uint2>([&](const int32_t i){return make_uint2(vec_len_v8, fmgr->GetFragLenById(i));}),
            decx::TArg_still<uint32_t>(pitchsrc),
            decx::TArg_still<__m256>(mask)
        );
    }
}