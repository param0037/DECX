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
HouseHolder_SingleCol_v4_fp64(const double* __restrict p_col,
                              double* __restrict       p_V,
                              const uint32_t          proc_len_v4,
                              const uint32_t          local_col_id,
                              const __m256d           front_mask)
{
    int32_t rval = 0;
    const uint32_t L_front = local_col_id % 4;
    const double x0 = p_col[L_front];

    // Calculate norm2 ^ 2
    double pow2_sum_post = 0;
    __m256d sum_v4 = _mm256_setzero_pd();
    for (int i = 0; i < proc_len_v4; ++i) {
        __m256d eles_v4 = _mm256_load_pd(p_col + (i << 2));
        if (i == 0) {
            eles_v4 = _mm256_and_pd(eles_v4, front_mask);
        }
        sum_v4 = _mm256_fmadd_pd(eles_v4, eles_v4, sum_v4);
    }
    pow2_sum_post = decx::utils::simd::_mm256d_h_sum(sum_v4);

    double x_norm2 = sqrt(pow2_sum_post);

    double sign = x0 < 0.0 ? 1.0 : -1.0;
    pow2_sum_post -= x0 * x0;
    double v0 = p_col[L_front] - sign * x_norm2;
    pow2_sum_post += v0 * v0;
    pow2_sum_post = sqrt(pow2_sum_post);

    const double _1_pow2_sum_post = 1.f / pow2_sum_post;
    for (int i = 0; i < proc_len_v4; ++i) {
        __m256d eles_v4 = _mm256_load_pd(p_col + (i << 2));
        eles_v4 = _mm256_mul_pd(eles_v4, _mm256_set1_pd(_1_pow2_sum_post));
        _mm256_store_pd(p_V + (i << 2), eles_v4);
    }
    p_V[L_front] = v0 / pow2_sum_post;
    __m256d eles_v4 = _mm256_load_pd(p_V);
    _mm256_store_pd(p_V, _mm256_and_pd(front_mask, eles_v4));
}


_THREAD_FUNCTION_ static void
Apply_Reflectors_v4_fp64(const double* __restrict    Vk, 
                         double* __restrict          panel_next, 
                         const uint32_t             local_col_id,
                         const uint2                proc_dims_v4,
                         const uint32_t             mat_pitch,
                         const __m256d               front_mask)
{
// #pragma omp parallel for
    for (int i = 0; i < proc_dims_v4.y; ++i) {
        __m256d sum_v4 = _mm256_setzero_pd();
        double* next_panel_col = panel_next + mat_pitch * i;
        for (int k = 0; k < proc_dims_v4.x; ++k) {
            __m256d vk_v4 = _mm256_load_pd(Vk + (k << 2));
            __m256d AR_v4 = _mm256_load_pd(next_panel_col + (k << 2));
            if (k == 0) {
                AR_v4 = _mm256_and_pd(AR_v4, front_mask);
            }
            sum_v4 = _mm256_fmadd_pd(vk_v4, AR_v4, sum_v4);
        }
        double res = decx::utils::simd::_mm256d_h_sum(sum_v4);
        res *= -2;
        for (int k = 0; k < proc_dims_v4.x; ++k) {
            __m256d vk_v4 = _mm256_load_pd(Vk + (k << 2));
            __m256d AR_v4 = _mm256_load_pd(next_panel_col + (k << 2));
            if (k == 0) {
                AR_v4 = _mm256_and_pd(AR_v4, front_mask);
            }
            AR_v4 = _mm256_fmadd_pd(_mm256_set1_pd(res), vk_v4, AR_v4);
            _mm256_store_pd(next_panel_col + (k << 2), AR_v4);
        }
    }
}

}
}
}


template <>
void decx::blas::Blocked_GQR_planner<double>::
sColHouseHolderTF(decx::blas::Blocked_GQR_planner<double>* _fake_this, 
                  const uint32_t                          local_col_id)
{
    const uint32_t proc_len_v1 = _fake_this->_block_dims.y - local_col_id;
    const uint32_t proc_len_v4 = decx::blas::Blocked_GQR_planner<double>::CalcProcLenV(local_col_id, 4, proc_len_v1);
    if (proc_len_v1 == _fake_this->_block_dims.x && local_col_id == _fake_this->_block_dims.x - 1){
        return;
    }
    __m256d mask = _mm256_setzero_pd();
    const uint32_t L_front = local_col_id % 4;
    _fake_this->GetPostMask(L_front, (void*)(&mask));

    decx::blas::CPUK::HouseHolder_SingleCol_v4_fp64(_fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_src, local_col_id, local_col_id),
        _fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_V, local_col_id, local_col_id),
        proc_len_v4, local_col_id, mask);
}


template <> void decx::blas::Blocked_GQR_planner<double>::
sApplyReflectors(decx::blas::Blocked_GQR_planner<double>* fake_this, 
                 const uint32_t local_col_id)
{
    int32_t rval = 0;
    const uint32_t L_front = local_col_id % 4;
    const uint32_t vec_len_v1 = fake_this->_block_dims.y - local_col_id;
    const uint32_t vec_len_v4 = CalcProcLenV(local_col_id, 4, vec_len_v1);
    const uint32_t pitchsrc = fake_this->_src_tile.GetDims().x;

    __m256d mask = _mm256_setzero_pd();
    rval |= fake_this->GetPostMask(L_front, (void*)(&mask));

    if (local_col_id < fake_this->_block_dims.x - 1) 
    {
        const decx::utils::frag_manager* fmgr = fake_this->_fmgrs_apply_HH + local_col_id;
        decx::utils::Thr1D t1D(fmgr->GetFragNum());
        
        const double* pV = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_V, local_col_id, local_col_id);
        double* pPanel = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_src, local_col_id, local_col_id + 1);

        decx::cpu_ElementWise1D_planner::
        sCaller(decx::blas::CPUK::Apply_Reflectors_v4_fp64, fmgr, &t1D, 
            decx::cpu::ThreadDispatchMethod_e::Dispatch_ByID,
            EW_SLOT_ID_MONOTONIC(0),
            decx::TArg_still<const double*>(pV),
            decx::TArg_var<double*>      ([&](const int32_t i){return pPanel + i * fmgr->GetFragLenById(0) * pitchsrc;}),
            decx::TArg_still<int32_t>(local_col_id),
            decx::TArg_var<uint2>([&](const int32_t i){return make_uint2(vec_len_v4, fmgr->GetFragLenById(i));}),
            decx::TArg_still<uint32_t>(pitchsrc),
            decx::TArg_still<__m256d>(mask)
        );
    }
}