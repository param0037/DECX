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
#include "../GEMM_callers.h"
#include "../matrix_B_arrange.h"
#include "GEMM_fp32_kernels.h"


decx::ResourceHandle decx::blas::g_cpu_GEMM_fp32_planner;


template <bool _ABC>
void decx::blas::GEMM_fp32_caller(const float*                      A, 
                                  const float*                      B, 
                                  float*                            dst, 
                                  const decx::_matrix_layout*       layout_A,
                                  const decx::_matrix_layout*       layout_dst, 
                                  const uint32_t                    Llen,
                                  const decx::utils::frag_manager*  f_mgrWH,
                                  const decx::blas::GEMM_blocking_config* _thread_configs, 
                                  decx::utils::_thr_2D*             t2D,
                                  const float*                      C)
{
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

            t2D->_async_thread[t2D->thread_w * i + j] = decx::cpu::register_task_default(
                decx::blas::CPUK::GEMM_fp32_kernel<_ABC>, A_loc, B_loc, dst_loc, conf_ptr,
                layout_A->pitch, conf_ptr->_fmgr_L.total, layout_dst->pitch, C_loc);

            B_loc += f_mgrWH[0].frag_len * Llen * 8;
            dst_loc += f_mgrWH[0].frag_len * 8;
            if constexpr (_ABC) { C_loc += f_mgrWH[0].frag_len * 8; }
        }

        const auto* conf_ptr = &_thread_configs[t2D->thread_w * (i + 1) - 1];

        t2D->_async_thread[t2D->thread_w * (i+1) - 1] = decx::cpu::register_task_default(
            decx::blas::CPUK::GEMM_fp32_kernel<_ABC>, A_loc, B_loc, dst_loc, conf_ptr,
            layout_A->pitch, conf_ptr->_fmgr_L.total, layout_dst->pitch, C_loc);

        A_loc += f_mgrWH[1].frag_len * layout_A->pitch;
    }

    t2D->__sync_all_threads();
}

template void decx::blas::GEMM_fp32_caller<true>(const float*, const float*, float*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*,
    const decx::blas::GEMM_blocking_config*, decx::utils::_thr_2D*, const float*);

template void decx::blas::GEMM_fp32_caller<false>(const float*, const float*, float*, const decx::_matrix_layout*,
    const decx::_matrix_layout*, const uint32_t, const decx::utils::frag_manager*,
    const decx::blas::GEMM_blocking_config*, decx::utils::_thr_2D*, const float*);




template <> template <>
void decx::blas::cpu_GEMM_planner<float>::Run<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst,
    decx::utils::_thread_arrange_2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_fp32((float*)B->Mat.ptr, 
        (float*)this->_arranged_B._ptr.ptr,
        B->Pitch(), 
        B->Height(), this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_fp32_caller<false>((float*)A->Mat.ptr, (float*)this->_arranged_B._ptr.ptr, 
        (float*)dst->Mat.ptr, this->_layout_A,
        &dst->get_layout(), A->Width(), this->_fmgr_WH_dst, this->_thread_config.ptr, t2D);
}



template <> template <>
void decx::blas::cpu_GEMM_planner<float>::Run<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst,
    decx::utils::_thread_arrange_2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_fp32((float*)B->Mat.ptr,
        (float*)this->_arranged_B._ptr.ptr,
        B->Pitch(),
        B->Height(), this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_fp32_caller<true>((float*)A->Mat.ptr, (float*)this->_arranged_B._ptr.ptr,
        (float*)dst->Mat.ptr, this->_layout_A,
        &dst->get_layout(), A->Width(), this->_fmgr_WH_dst, this->_thread_config.ptr, t2D, (float*)C->Mat.ptr);
}


template <bool _ABC>
void decx::blas::GEMM_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle,
    decx::_Matrix* C)
{
    if (decx::blas::g_cpu_GEMM_fp32_planner._res_ptr == NULL) {
        decx::blas::g_cpu_GEMM_fp32_planner.RegisterResource(new decx::blas::cpu_GEMM_planner<float>,
            5, &decx::blas::cpu_GEMM_planner<float>::Release);
    }

    decx::blas::g_cpu_GEMM_fp32_planner.lock();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();
    //const uint32_t _conc = 1;

    auto* _planner = decx::blas::g_cpu_GEMM_fp32_planner.get_resource_raw_ptr<decx::blas::cpu_GEMM_planner<float>>();

    // Validate the sizes of the matrices
    if constexpr (_ABC) {
        decx::blas::cpu_GEMM_planner<float>::Validate(handle, &A->get_layout(), &B->get_layout(), &C->get_layout());
    }
    else {
        decx::blas::cpu_GEMM_planner<float>::Validate(handle, &A->get_layout(), &B->get_layout());
    }
    Check_Runtime_Error(handle);

    // Plan if changed
    if (_planner->Changed(_conc, &A->get_layout(), &B->get_layout())) {
        _planner->plan(decx::cpu::_get_permitted_concurrency(), &A->get_layout(), &B->get_layout(), de::GetLastError());
        Check_Runtime_Error(handle);
    }

    decx::utils::_thread_arrange_2D t2D(_planner->GetThreadDist_B().y, _planner->GetThreadDist_B().x);
    if constexpr (_ABC) {
        _planner->Run<false>(A, B, C, dst, &t2D);
    }
    else {
        _planner->Run<false>(A, B, dst, &t2D);
    }

    decx::blas::g_cpu_GEMM_fp32_planner.unlock();
}

template void decx::blas::GEMM_fp32<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*, decx::_Matrix*);
template void decx::blas::GEMM_fp32<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*, decx::_Matrix*);
