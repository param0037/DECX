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
#include <Element_wise/common/cpu_element_wise_planner.h>
#define MODULE_TAG "GQR_cpu"


namespace decx
{
namespace blas
{
namespace CPUK
{
static void UpdateW_k1_v4_fp64(const double* __restrict pV, 
                               double* __restrict       pW, 
                               const uint32_t          proc_len_v1)
{
    const __m256d scalar = _mm256_set1_pd(2.0f);
    for (int32_t i = 0; i < decx::utils::idiv_ceil<uint32_t>(proc_len_v1, 4); ++i){
        decx::utils::simd::xmm256_reg Vval;
        Vval._vd = _mm256_load_pd((double*)(pV + (i << 2)));
        _mm256_store_pd((double*)(pW + (i << 2)), _mm256_mul_pd(Vval._vd, scalar));
    }
}


/**
 * @brief Calculate the IWY = I - W * V^T for k=1 iteration of the blocked GQR algorithm.
 * 
 * @param pV The V matrix.
 * @param pW The W matrix.
 * @param start_idx_WH The starting index of the W matrix.
 * @param proc_len_v1 The length of the V matrix.
 * @param alignment The alignment of the V matrix.
 */
template <bool IWY_initial>
_THREAD_FUNCTION_ static void 
CalcIWY_blocked_v4_fp64(const double* __restrict        pV_last, 
                        const double* __restrict        pW_last, 
                        double* __restrict              pIWY,
                        const uint2                    start_idx_WH,
                        const uint2                    proc_sizes_v4_WH,
                        const uint32_t                 pitch_IWY_v1,
                        const void*                    p_post_mask_v4)
{
    int2 g_coord_WH = make_int2(start_idx_WH.x, start_idx_WH.y);
    const __m256d post_mask_v4 = _mm256_loadu_pd((const double*)p_post_mask_v4);
    const double* p_mask = (double*)p_post_mask_v4;

    for (int32_t i = 0; i < proc_sizes_v4_WH.y; ++i)
    {
        // Load value from W as the scalar of this row.
        double Wval = pW_last[i];
        __m256d Wval_v4 = _mm256_set1_pd(Wval);
        double* pIWY_row = pIWY + i * pitch_IWY_v1;
        g_coord_WH.x = start_idx_WH.x;

        // printf("proc_sizes_v4_WH.x=%d, pV_last: [", proc_sizes_v4_WH.x);
        // for (int test = 0; test < 16; ++test) {
        //     printf("%lf, ", pV_last[test]);
        // }
        // printf("]\n");

        for (int32_t j = 0; j < proc_sizes_v4_WH.x; ++j)
        {
            __m256d Vval_v4 = _mm256_load_pd(pV_last + (j << 2));
            __m256d product_v4 = _mm256_mul_pd(Vval_v4, Wval_v4);

            decx::utils::simd::xmm256_reg IWY_v4;
            IWY_v4._vd = _mm256_setzero_pd();
            if constexpr (IWY_initial) {
                if ((g_coord_WH.y >> 2) == (g_coord_WH.x >> 2)){
                    IWY_v4._arrd[g_coord_WH.y % 4] = 1.0;
                }
            }
            else{
                IWY_v4._vd = _mm256_load_pd(pIWY_row + (j << 2));
            }
            if (j == 0) {       // Mask the first lane
                product_v4 = _mm256_and_pd(product_v4, post_mask_v4);
            }
            product_v4 = _mm256_sub_pd(IWY_v4._vd, product_v4);
            _mm256_store_pd(pIWY_row + (j << 2), product_v4);
            g_coord_WH.x += 4;
        }
        ++g_coord_WH.y;
    }
}

}
}
}


template <> void 
decx::blas::Blocked_GQR_planner<double>::sUpdateW(decx::blas::Blocked_GQR_planner<double>* fake_this,
                                                 const uint32_t                          local_col_id)
{
    const uint32_t alignment = fake_this->_align_bytes / sizeof(double);
    const uint32_t proc_len_v1 = fake_this->_block_dims.y - local_col_id;

    decx::utils::Thr1D t1D(fake_this->_fmgr_updateW.GetFragNum());
    const uint32_t pitchIWY = fake_this->_IWY.GetDims().x;
    
    if (local_col_id == 0) {
        const double* pV_now = fake_this->GetV() + 0;
        double* pW = (double*)fake_this->_W_tile + 0;
        CPUK::UpdateW_k1_v4_fp64(pV_now, pW, proc_len_v1);
    }
    else {
        uint8_t post_mask[32];
        fake_this->GetPostMask((local_col_id - 1) % 4, post_mask);
        auto* pFunc = local_col_id == 1 ? CPUK::CalcIWY_blocked_v4_fp64<true> : CPUK::CalcIWY_blocked_v4_fp64<false>;
        
        const double* pV = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_V, local_col_id - 1, local_col_id - 1);    // V(k-1:end, k-1)
        const double* pW = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_W, 0, local_col_id - 1);                   // W(:, k-1:end)
        double* pIWY = fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_IWY, local_col_id - 1, 0);                     // IWY(:, k-1:end)
        
        decx::cpu_ElementWise1D_planner::
            sCaller(pFunc, &fake_this->_fmgr_updateW, &t1D, 
                decx::cpu::ThreadDispatchMethod_e::Dispatch_ByID,
                EW_SLOT_ID_MONOTONIC(0),
                decx::TArg_still<const double*>(pV),
                decx::TArg_var<const double*>([&](const int32_t i){return pW + i * fake_this->_fmgr_updateW.GetFragLenById(0);}),
                decx::TArg_var<double*>([&](const int32_t i){return pIWY + i * pitchIWY * fake_this->_fmgr_updateW.GetFragLenById(0);}),
                decx::TArg_var<uint2>([&](const int32_t i){return make_uint2(0, i * fake_this->_fmgr_updateW.GetFragLenById(0));}),
                decx::TArg_var<uint2>([&](const int32_t i){return make_uint2(decx::utils::idiv_ceil<uint32_t>(proc_len_v1 + 1, 4), fake_this->_fmgr_updateW.GetFragLenById(i));}),
                decx::TArg_still<uint32_t>(pitchIWY),
                decx::TArg_still<void*>((void*)post_mask));
        
        // Update W
        fake_this->_w_update_helpers[(local_col_id - 1) / alignment].Run(
            fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_IWY, local_col_id - 1, 0),           // IWY(:, k-1:end)
            fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_V, local_col_id - 1, local_col_id),  // V(k-1:end, k)
            fake_this->GetAlignedBufAddr(BlockedGQR_BufType_e::BGQR_Buffer_W, 0, local_col_id),                 // W(:, k)
            fake_this->_IWY.GetDims().x, 2.0, 0);
    }
}
