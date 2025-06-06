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
    template <bool _ABC> static
    void GEMM_fp32_caller(const float* A, const float* B, float* dst, const decx::_matrix_layout* layout_A,
        const decx::_matrix_layout* layout_dst, const uint32_t Llen, const decx::utils::frag_manager *f_mgrH,
        const decx::blas::GEMM_blocking_config* _thread_configs, decx::utils::Thr2D* t1D, const float* C = NULL);
}
}


template <bool _ABC>
void decx::blas::GEMM_fp32_caller(const float*                      A, 
                                  const float*                      B, 
                                  float*                            dst, 
                                  const decx::_matrix_layout*       layout_A,
                                  const decx::_matrix_layout*       layout_dst, 
                                  const uint32_t                    Llen,
                                  const decx::utils::frag_manager*  f_mgrWH,
                                  const decx::blas::GEMM_blocking_config* _thread_configs, 
                                  decx::utils::Thr2D*             t2D,
                                  const float*                      C)
{
#if defined(__x86_64__) || defined(__i386__)
    constexpr uint32_t _alignment = 8;
#endif
#if defined(__aarch64__) || defined(__arm__)
    constexpr uint32_t _alignment = 4;
#endif

    const float* A_loc = A;
    const float* B_loc = B;
    float* dst_loc = dst;
    const float* C_loc = C;

    for (uint32_t i = 0; i < t2D->thread_h; ++i) 
    {
        B_loc = B;
        dst_loc = dst + i * layout_dst->pitch * f_mgrWH[1].frag_len;

        for (uint32_t j = 0; j < t2D->thread_w - 1; ++j) 
        {
            const auto* conf_ptr = &_thread_configs[t2D->thread_w * i + j];

            t2D->_async_thread[t2D->thread_w * i + j] = decx::cpu::RegisterTaskLoadBalanced(
                decx::blas::CPUK::GEMM_fp32_kernel<_ABC>, A_loc, B_loc, dst_loc, conf_ptr,
                layout_A->pitch, conf_ptr->_fmgr_L.total, layout_dst->pitch, C_loc);

            B_loc += f_mgrWH[0].frag_len * Llen * _alignment;
            dst_loc += f_mgrWH[0].frag_len * _alignment;
            if constexpr (_ABC) { C_loc += f_mgrWH[0].frag_len * _alignment; }
        }

        const auto* conf_ptr = &_thread_configs[t2D->thread_w * (i + 1) - 1];

        t2D->_async_thread[t2D->thread_w * (i+1) - 1] = decx::cpu::RegisterTaskLoadBalanced(
            decx::blas::CPUK::GEMM_fp32_kernel<_ABC>, A_loc, B_loc, dst_loc, conf_ptr,
            layout_A->pitch, conf_ptr->_fmgr_L.total, layout_dst->pitch, C_loc);

        A_loc += f_mgrWH[1].frag_len * layout_A->pitch;
    }

    t2D->__sync_all_threads();
}

template void decx::blas::GEMM_fp32_caller<true>(const float*, const float*, float*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*,
    const decx::blas::GEMM_blocking_config*, decx::utils::Thr2D*, const float*);

template void decx::blas::GEMM_fp32_caller<false>(const float*, const float*, float*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*,
    const decx::blas::GEMM_blocking_config*, decx::utils::Thr2D*, const float*);




template <> template <>
void decx::blas::cpu_GEMM_planner<float>::Run<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst,
    decx::utils::ThreadArrange2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_fp32(B->Mat.GetRawPtr<float>(), 
        this->_arranged_B.GetRawPtr<float>(),
        // dst->Mat.GetRawPtr<float>(),
        B->Pitch(), 
        B->Height(), this->_fmgr_WH_B, t2D);
        
    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);
    
    // Execute GEMM
    decx::blas::GEMM_fp32_caller<false>(A->Mat.GetRawPtr<float>(),          this->_arranged_B.GetRawPtr<float>(), 
                                        dst->Mat.GetRawPtr<float>(),        this->_layout_A,
                                        &dst->get_layout(), A->Width(),     this->_fmgr_WH_dst, 
                                        this->_thread_config.GetRawPtr(),   t2D);
}


template <> template <>
void decx::blas::cpu_GEMM_planner<float>::Run<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst,
    decx::utils::ThreadArrange2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_fp32(B->Mat.GetRawPtr<float>(),
        this->_arranged_B.GetRawPtr<float>(),
        B->Pitch(),
        B->Height(), this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_fp32_caller<true>(A->Mat.GetRawPtr<float>(),       this->_arranged_B.GetRawPtr<float>(),
                                       dst->Mat.GetRawPtr<float>(),     this->_layout_A,
                                       &dst->get_layout(), A->Width(),  this->_fmgr_WH_dst, this->_thread_config.GetRawPtr(), 
                                       t2D,                             C->Mat.GetRawPtr<float>());
}
