/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../cpu_GEMM_config.h"
#include "../GEMM_callers.h"
#include "../matrix_B_arrange.h"


decx::ResourceHandle decx::blas::g_cpu_GEMM_fp32_planner;


template <>
void decx::blas::cpu_GEMM_planner<float>::Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst,
    decx::utils::_thread_arrange_2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_fp32((float*)B->Mat.ptr, (float*)this->_arranged_B._ptr.ptr, B->Pitch(), this->_arranged_B._dims.x / 8, this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_fp32_caller<false>((float*)A->Mat.ptr, (float*)this->_arranged_B._ptr.ptr, 
        (float*)dst->Mat.ptr, this->_layout_A,
        &dst->get_layout(), this->_arranged_B._dims, this->_fmgr_WH_dst, this->_thread_config.ptr, t2D);
}



template <>
void decx::blas::cpu_GEMM_planner<float>::Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst,
    decx::utils::_thread_arrange_2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_fp32((float*)B->Mat.ptr, (float*)this->_arranged_B._ptr.ptr, B->Pitch(), this->_arranged_B._dims.x / 8, this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_fp32_caller<true>((float*)A->Mat.ptr, (float*)this->_arranged_B._ptr.ptr,
        (float*)dst->Mat.ptr, this->_layout_A,
        &dst->get_layout(), this->_arranged_B._dims, this->_fmgr_WH_dst, this->_thread_config.ptr, t2D, (float*)C->Mat.ptr);
}
