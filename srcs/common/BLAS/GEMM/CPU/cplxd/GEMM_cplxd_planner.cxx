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
    template <bool _ABC> static
        void GEMM_cplxd_caller(const de::CPd* A, const de::CPd* B, de::CPd* dst, const decx::_matrix_layout* layout_A,
            const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager* f_mgrH,
            const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::Thr2D* t1D, const de::CPd* C = NULL);
}
}


template <bool _ABC> void 
decx::blas::GEMM_cplxd_caller(const de::CPd* A,                            const de::CPd* B, 
                              de::CPd* dst,                                const decx::_matrix_layout* layout_A,
                              const decx::_matrix_layout* layout_dst,      const uint32_t Llen,
                              const decx::utils::frag_manager* f_mgrWH,    const decx::blas::GEMM_blocking_config* _thread_configs, 
                              decx::utils::Thr2D* t2D,                   const de::CPd* C)
{
    const de::CPd* A_loc = A;
    const de::CPd* B_loc = B;
    de::CPd* dst_loc = dst;
    const de::CPd* C_loc = C;

    for (uint32_t i = 0; i < t2D->thread_h; ++i) 
    {
        B_loc = B;
        dst_loc = dst + i * layout_dst->pitch * f_mgrWH[1].frag_len;
        C_loc = C + i * layout_dst->pitch * f_mgrWH[1].frag_len;

        for (uint32_t j = 0; j < t2D->thread_w - 1; ++j) 
        {
            const auto* conf_ptr = &_thread_configs[t2D->thread_w * i + j];

            t2D->_async_thread[t2D->thread_w * i + j] = decx::cpu::RegisterTaskLoadBalanced(
                decx::blas::CPUK::GEMM_cplxd_kernel<_ABC>, A_loc, B_loc, dst_loc, conf_ptr,
                layout_A->pitch, conf_ptr->_fmgr_L.total, layout_dst->pitch, C_loc);

            B_loc += f_mgrWH[0].frag_len * Llen * 2;
            dst_loc += f_mgrWH[0].frag_len * 2;
            if constexpr (_ABC) { C_loc += f_mgrWH[0].frag_len * 2; }
        }

        const auto* conf_ptr = &_thread_configs[t2D->thread_w * (i + 1) - 1];

        t2D->_async_thread[t2D->thread_w * (i + 1) - 1] = decx::cpu::RegisterTaskLoadBalanced(
            decx::blas::CPUK::GEMM_cplxd_kernel<_ABC>, A_loc, B_loc, dst_loc, conf_ptr,
            layout_A->pitch, conf_ptr->_fmgr_L.total, layout_dst->pitch, C_loc);

        A_loc += f_mgrWH[1].frag_len * layout_A->pitch;
    }

    t2D->__sync_all_threads();
}

template void decx::blas::GEMM_cplxd_caller<true>(const de::CPd*, const de::CPd*, de::CPd*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*,
    const decx::blas::GEMM_blocking_config*, decx::utils::Thr2D*, const de::CPd*);

template void decx::blas::GEMM_cplxd_caller<false>(const de::CPd*, const de::CPd*, de::CPd*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*,
    const decx::blas::GEMM_blocking_config*, decx::utils::Thr2D*, const de::CPd*);



template <> template <>
void decx::blas::cpu_GEMM_planner<de::CPd>::Run<true>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst,
    decx::utils::ThreadArrange2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_cplxd(B->Mat.GetRawPtr<de::CPd>(), 
                                       this->_arranged_B.GetRawPtr<de::CPd>(),
                                       B->Pitch(), 
                                       B->Height(), this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_cplxd_caller<false>(A->Mat.GetRawPtr<de::CPd>(),       this->_arranged_B.GetRawPtr<de::CPd>(), 
                                          dst->Mat.GetRawPtr<de::CPd>(),    this->_layout_A,
                                          &dst->get_layout(),               A->Width(), 
                                          this->_fmgr_WH_dst,               this->_thread_config.GetRawPtr(), t2D);
}


template <> template <>
void decx::blas::cpu_GEMM_planner<de::CPd>::Run<true>(decx::_Matrix* A, decx::_Matrix* B, 
    decx::_Matrix* C, decx::_Matrix* dst, decx::utils::ThreadArrange2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_cplxd(B->Mat.GetRawPtr<de::CPd>(),
                                       this->_arranged_B.GetRawPtr<de::CPd>(),
                                       B->Pitch(),
                                       B->Height(), this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_cplxd_caller<true>(A->Mat.GetRawPtr<de::CPd>(),        this->_arranged_B.GetRawPtr<de::CPd>(),
                                        dst->Mat.GetRawPtr<de::CPd>(),      this->_layout_A,
                                        &dst->get_layout(), A->Width(),     this->_fmgr_WH_dst, 
                                        this->_thread_config.GetRawPtr(),   t2D, C->Mat.GetRawPtr<de::CPd>());
}
