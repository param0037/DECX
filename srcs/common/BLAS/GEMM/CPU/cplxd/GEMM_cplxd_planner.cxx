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
#include "GEMM_cplxd_kernels.h"


namespace decx
{
namespace blas
{
    template <bool _ABC> static int32_t GEMM_cplxd_caller(const de::CPd* A, const de::CPd* B, de::CPd* dst, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager* f_mgrH, 
        const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::ComputeLoadsMgr2D* t1D, const de::CPd* C = NULL);
}
}


template <bool _ABC> int32_t 
decx::blas::GEMM_cplxd_caller(const de::CPd* A,                             const de::CPd* B, 
                              de::CPd* dst,                                 const decx::_matrix_layout* layout_A,
                              const decx::_matrix_layout* layout_dst,       const uint32_t Llen,
                              const decx::utils::frag_manager* f_mgrWH,     const decx::blas::GEMM_blocking_config* _thread_configs, 
                              decx::utils::ComputeLoadsMgr2D* t2D,          const de::CPd* C)
{
    int32_t rval = 0;

    const de::CPd* A_loc = A;
    const de::CPd* B_loc = B;
    de::CPd* dst_loc = dst;
    const de::CPd* C_loc = C;

    uint32_t task_id = 0;
    for (uint32_t i = 0; i < t2D->GetDist().y; ++i) 
    {
        B_loc = B;
        dst_loc = dst + i * layout_dst->pitch * f_mgrWH[1].frag_len;
        C_loc = C + i * layout_dst->pitch * f_mgrWH[1].frag_len;

        for (uint32_t j = 0; j < t2D->GetDist().x; ++j) 
        {
            const auto* conf_ptr = &_thread_configs[task_id];

            rval |= t2D->AppendTask(task_id, decx::blas::CPUK::GEMM_cplxd_kernel<_ABC>, 
                PACK_CPY(A_loc), PACK_CPY(B_loc), PACK_CPY(dst_loc), PACK_CPY(conf_ptr), PACK_REF(layout_A->pitch), 
                PACK_REF(conf_ptr->_fmgr_L.total), PACK_REF(layout_dst->pitch), PACK_CPY(C_loc));

            B_loc += f_mgrWH[0].frag_len * Llen * 2;
            dst_loc += f_mgrWH[0].frag_len * 2;
            if_opt (_ABC) { C_loc += f_mgrWH[0].frag_len * 2; }
            ++task_id;
        }
        A_loc += f_mgrWH[1].frag_len * layout_A->pitch;
    }

    rval |= t2D->RunAll();
    rval |= t2D->SynchronizeAll();
    rval |= t2D->ClearAll();

    return rval;
}

template int32_t decx::blas::GEMM_cplxd_caller<true>(const de::CPd*, const de::CPd*, de::CPd*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*, const decx::blas::GEMM_blocking_config*, 
    decx::utils::ComputeLoadsMgr2D*, const de::CPd*);

template int32_t decx::blas::GEMM_cplxd_caller<false>(const de::CPd*, const de::CPd*, de::CPd*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*, const decx::blas::GEMM_blocking_config*, 
    decx::utils::ComputeLoadsMgr2D*, const de::CPd*);



template <> template <>
int32_t decx::blas::cpu_GEMM_planner<de::CPd>::Run<true>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst)
{
    int32_t rval = 0;
    rval |= this->_tasks.Reshape(this->GetThreadDist_B());
    // Arrange matrix B
    rval |= decx::blas::matrix_B_arrange_cplxd(B->Mat.GetRawPtr<de::CPd>(), 
                                               this->_arranged_B.GetRawPtr<de::CPd>(),
                                               B->Pitch(), 
                                               B->Height(), this->_fmgr_WH_B, &this->_tasks);

    // Reshape to adapt the thread distribution of kernels
    rval |= this->_tasks.Reshape(this->GetThreadDist_dst());

    // Execute GEMM
    rval |= decx::blas::GEMM_cplxd_caller<false>(
        A->Mat.GetRawPtr<de::CPd>(),    this->_arranged_B.GetRawPtr<de::CPd>(), 
        dst->Mat.GetRawPtr<de::CPd>(),  this->_layout_A,
        &dst->get_layout(),             A->Width(), 
        this->_fmgr_WH_dst,             this->_thread_config.GetRawPtr(), 
        &this->_tasks);
    
    return rval;
}


template <> template <>
int32_t decx::blas::cpu_GEMM_planner<de::CPd>::Run<true>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst)
{
    int32_t rval = 0;
    rval |= this->_tasks.Reshape(this->GetThreadDist_B());
    // Arrange matrix B
    rval |= decx::blas::matrix_B_arrange_cplxd(B->Mat.GetRawPtr<de::CPd>(),
                                               this->_arranged_B.GetRawPtr<de::CPd>(),
                                               B->Pitch(),
                                               B->Height(), this->_fmgr_WH_B, &this->_tasks);

    // Reshape to adapt the thread distribution of kernels
    rval |= this->_tasks.Reshape(this->GetThreadDist_dst());

    // Execute GEMM
    rval |= decx::blas::GEMM_cplxd_caller<true>(
        A->Mat.GetRawPtr<de::CPd>(),        this->_arranged_B.GetRawPtr<de::CPd>(),
        dst->Mat.GetRawPtr<de::CPd>(),      this->_layout_A,
        &dst->get_layout(), A->Width(),     this->_fmgr_WH_dst, 
        this->_thread_config.GetRawPtr(),   &this->_tasks, C->Mat.GetRawPtr<de::CPd>());
    
    return rval;
}
