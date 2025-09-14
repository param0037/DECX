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


#include "../cpu_GEMM_config.h"
#include "../matrix_B_arrange.h"
#if defined(__x86_64__) || defined(__i386__)
#include "x86/GEMM_fp32_kernels_x86_64.h"
#endif
#if defined(__aarch64__) || defined(__arm__)
#include "arm/GEMM_fp32_kernels_aarch64.h"
#endif


namespace decx
{
namespace blas
{
    template <bool _ABC> static int32_t GEMM_fp32_caller(const float* A, const float* B, float* dst, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager *f_mgrH, 
        const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::ComputeLoadsMgr2D* t1D, const float* C = NULL);
}
}


template <bool _ABC>
int32_t decx::blas::GEMM_fp32_caller(const float* A,                           const float* B, 
                                     float* dst,                               const decx::_matrix_layout* layout_A,
                                     const decx::_matrix_layout* layout_dst,   const uint32_t Llen,
                                     const decx::utils::frag_manager* f_mgrWH, const decx::blas::GEMM_blocking_config* _thread_configs, 
                                     decx::utils::ComputeLoadsMgr2D* t2D,      const float* C)
{
    int32_t rval = 0;

    constexpr uint32_t _alignment = decx::utils::simd::GetCPUSimdAlignBytes() / sizeof(float);

    const float* A_loc = A;
    const float* B_loc = B;
    float* dst_loc = dst;
    const float* C_loc = C;

    uint32_t task_id = 0;

    for (uint32_t i = 0; i < t2D->GetDist().y; ++i) 
    {
        B_loc = B;
        dst_loc = dst + i * layout_dst->pitch * f_mgrWH[1].frag_len;

        for (uint32_t j = 0; j < t2D->GetDist().x; ++j) 
        {
            const auto* conf_ptr = &_thread_configs[task_id];
            rval |= t2D->AppendTask(task_id, decx::blas::CPUK::GEMM_fp32_kernel<_ABC>, 
                PACK_CPY(A_loc), PACK_CPY(B_loc), PACK_CPY(dst_loc), PACK_CPY(conf_ptr),
                PACK_REF(layout_A->pitch), PACK_REF(conf_ptr->_fmgr_L.total), PACK_REF(layout_dst->pitch), PACK_CPY(C_loc));

            B_loc += f_mgrWH[0].frag_len * Llen * _alignment;
            dst_loc += f_mgrWH[0].frag_len * _alignment;
            if_opt (_ABC) { C_loc += f_mgrWH[0].frag_len * _alignment; }
            ++task_id;
        }
        
        A_loc += f_mgrWH[1].frag_len * layout_A->pitch;
    }

    rval |= t2D->RunAll();
    rval |= t2D->SynchronizeAll();
    rval |= t2D->ClearAll();

    return rval;
}

template int32_t decx::blas::GEMM_fp32_caller<true>(const float*, const float*, float*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*, const decx::blas::GEMM_blocking_config*, 
    decx::utils::ComputeLoadsMgr2D*, const float*);

template int32_t decx::blas::GEMM_fp32_caller<false>(const float*, const float*, float*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*, const decx::blas::GEMM_blocking_config*, 
    decx::utils::ComputeLoadsMgr2D*, const float*);



template <> template <>
int32_t decx::blas::cpu_GEMM_planner<float>::Run<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst)
{
    int32_t rval = 0;
    rval |= this->_tasks.Reshape(this->GetThreadDist_B());
    // Arrange matrix B
    rval |= decx::blas::matrix_B_arrange_fp32(B->Mat.GetRawPtr<float>(), 
                                              this->_arranged_B.GetRawPtr<float>(),
                                              B->Pitch(), 
                                              B->Height(), this->_fmgr_WH_B, &this->_tasks);
        
    // Reshape to adapt the thread distribution of kernels
    rval |= this->_tasks.Reshape(this->GetThreadDist_dst());
    
    // Execute GEMM
    rval |= decx::blas::GEMM_fp32_caller<false>(
        A->Mat.GetRawPtr<float>(),          this->_arranged_B.GetRawPtr<float>(), 
        dst->Mat.GetRawPtr<float>(),        this->_layout_A,
        &dst->get_layout(),                 A->Width(),
        this->_fmgr_WH_dst,                 this->_thread_config.GetRawPtr(),
        &this->_tasks);
    
    return rval;
}


template <> template <>
int32_t decx::blas::cpu_GEMM_planner<float>::Run<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst)
{
    int32_t rval = 0;
    rval |= this->_tasks.Reshape(this->GetThreadDist_B());
    // Arrange matrix B
    rval |= decx::blas::matrix_B_arrange_fp32(B->Mat.GetRawPtr<float>(),
                                              this->_arranged_B.GetRawPtr<float>(),
                                              B->Pitch(),
                                              B->Height(), this->_fmgr_WH_B, &this->_tasks);

    // Reshape to adapt the thread distribution of kernels
    rval |= this->_tasks.Reshape(this->GetThreadDist_dst());

    // Execute GEMM
    rval |= decx::blas::GEMM_fp32_caller<true>(
        A->Mat.GetRawPtr<float>(),      this->_arranged_B.GetRawPtr<float>(),
        dst->Mat.GetRawPtr<float>(),    this->_layout_A,
        &dst->get_layout(),             A->Width(),
        this->_fmgr_WH_dst,             this->_thread_config.GetRawPtr(), 
        &this->_tasks,                  C->Mat.GetRawPtr<float>());
    
    return rval;
}
