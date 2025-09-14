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

#include "MVM_execs.h"
#include <SIMD/intrinsics_ops.h>
#include "MVM_planner.h"
#define MODULE_TAG "MVM_cpu"


namespace decx
{
namespace blas
{
namespace CPUK
{
    _THREAD_CALL_ static void 
    MVM_blocked_v8_fp32(const float* __restrict mat, const float* __restrict vec, float* __restrict res_vec, const uint32_t proc_L_v1, const uint32_t proc_H_v1, 
        const uint32_t mat_pitch, const float alpha = 1.f, const float c = 0.f, const void* mask = NULL);


    _THREAD_FUNCTION_ static void
    MVM_exec_v8_fp32(const float* __restrict mat, const float* __restrict vec, float* __restrict res_vec, const decx::utils::frag_manager* block_conf_L, 
        const decx::utils::frag_manager* block_conf_H, const uint32_t mat_pitch, const float alpha = 1.f, const float c = 0.f, const void* mask = NULL);
}
}
}


_THREAD_CALL_ void
decx::blas::CPUK::MVM_blocked_v8_fp32(const float* __restrict   mat, 
                                      const float* __restrict   vec, 
                                      float* __restrict         res_vec, 
                                      const uint32_t            proc_L_v1, 
                                      const uint32_t            proc_H_v1, 
                                      const uint32_t            mat_pitch,
                                      const float               alpha,
                                      const float               c,
                                      const void*               mask)
{
    uint32_t dex_mat = 0;
    decx::utils::simd::xmm256_reg mask_v8;
    const uint32_t proclen_w_v8 = decx::utils::idiv_ceil<uint32_t>(proc_L_v1, 8);

    __m256 alpha_v8, c_v8;
    const bool use_alpha = fabs(alpha - 1.f) > 1e-9f;
    if (use_alpha){
        alpha_v8 = _mm256_set1_ps(alpha);
    }
    const bool use_c = fabs(c - 0.f) > 1e-9f;
    if (use_c) {
        c_v8 = _mm256_set1_ps(c);
    }

    if (mask != NULL) mask_v8._vf = _mm256_loadu_ps((float*)mask);
    else mask_v8._vi = _mm256_set1_epi32(0xFFFFFFFF);

    for (int32_t i = 0; i < proc_H_v1; i++) 
    {
        decx::utils::simd::xmm256_reg res_v8;
        res_v8._vf = _mm256_setzero_ps();
        dex_mat = i * mat_pitch;
        for (int32_t j = 0; j < proclen_w_v8; j++) {
            // Load the vector value
            decx::utils::simd::xmm256_reg vecval_v8;
            vecval_v8._vf = _mm256_loadu_ps(vec + j * 8);
            // Load the matrix value
            decx::utils::simd::xmm256_reg matval_v8;
            matval_v8._vf = _mm256_loadu_ps(mat + dex_mat);
            
            if (j == proclen_w_v8 - 1) {
                vecval_v8._vf = _mm256_and_ps(vecval_v8._vf, mask_v8._vf);
                matval_v8._vf = _mm256_and_ps(matval_v8._vf, mask_v8._vf);
            }
            
            // Accumulate the result
            if (use_alpha){
                matval_v8._vf = _mm256_mul_ps(matval_v8._vf, alpha_v8);
            }
            res_v8._vf = _mm256_fmadd_ps(matval_v8._vf, vecval_v8._vf, res_v8._vf);
            if (use_alpha){
                res_v8._vf = _mm256_add_ps(res_v8._vf, c_v8);
            }
            dex_mat += 8;
        }
        res_vec[i] += decx::utils::simd::_mm256_h_sum(res_v8._vf);
    }
}


_THREAD_FUNCTION_ void
decx::blas::CPUK::MVM_exec_v8_fp32(const float* __restrict   mat, 
                                   const float* __restrict   vec, 
                                   float* __restrict         res_vec, 
                                   const decx::utils::frag_manager* block_conf_L, 
                                   const decx::utils::frag_manager* block_conf_H, 
                                   const uint32_t            mat_pitch,
                                   const float               alpha,
                                   const float               c,
                                   const void*               mask)
{
    const float* p_mat_block = mat;
    for (int32_t i = 0; i < block_conf_H->frag_num; i++) 
    {
        const float* p_mat_block = mat + i * mat_pitch * block_conf_H->GetFragLen();
        const float* p_vec = vec;
        float* p_res = res_vec + i * block_conf_H->GetFragLen();
        for (int32_t j = 0; j < block_conf_L->frag_num; j++) 
        {
            MVM_blocked_v8_fp32(p_mat_block, p_vec, p_res, block_conf_L->GetFragLenById(j), block_conf_H->GetFragLenById(i), mat_pitch,
                alpha, c, j < block_conf_L->frag_num - 1 ? nullptr : mask);
            p_mat_block += block_conf_L->GetFragLen();
            p_vec += block_conf_L->GetFragLen();
        }
    }
}


template <> int32_t 
decx::blas::cpu_MVM_planner<float>::Run(const float* __restrict mat, 
                                        const float* __restrict vec, 
                                        float* __restrict res_vec, 
                                        const uint32_t mat_pitch, 
                                        const float alpha, 
                                        const float c)
{
    if (nullptr == this->_task_mgr){
        DECX_LOG_ERR("Task mgr not hooked");
        return -1;
    }
    this->_task_mgr->SetMaxThreadNum(this->_concurrency);
    this->_task_mgr->SetDispatchMethod(decx::core::ThreadDispatchMethod_e::Dispatch_ByID);

    decx::cpu_ElementWise1D_planner::sCaller(
            decx::blas::CPUK::MVM_exec_v8_fp32, 
            &this->_fmgr_H, this->_task_mgr,
            EW_SLOT_ID_MONOTONIC(0),
            decx::TArg_var<const float*>([&](const int32_t i){return mat + mat_pitch * i * this->_fmgr_H.GetFragLen();}),
            decx::TArg_still<const float*>(vec),
            decx::TArg_var<float*>([&](const int32_t i){return res_vec + i * this->_fmgr_H.GetFragLen();}),
            decx::TArg_still<const decx::utils::frag_manager*>(&this->_fmgr_L),
            decx::TArg_var<const decx::utils::frag_manager*>([&](const int32_t i){return this->_block_confs_H_perthread + i;}),
            decx::TArg_still<uint32_t>(mat_pitch),
            decx::TArg_still<float>(alpha),
            decx::TArg_still<float>(c),
            decx::TArg_still<const void*>(this->_L_align_mask));

    return 0;
}
