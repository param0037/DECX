


#include "../cpu_GEMM_config.h"
#include "../GEMM_callers.h"
#include "../matrix_B_arrange.h"


decx::ResourceHandle decx::blas::g_cpu_GEMM_64b_planner;


template <>
void decx::blas::cpu_GEMM_planner<double>::Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst,
    decx::utils::_thread_arrange_2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_64b((double*)B->Mat.ptr, (double*)this->_arranged_B._ptr.ptr, B->Pitch(), this->_arranged_B._dims.x / 4, this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_64b_caller<false>((double*)A->Mat.ptr, (double*)this->_arranged_B._ptr.ptr,
        (double*)dst->Mat.ptr, this->_layout_A,
        &dst->get_layout(), this->_arranged_B._dims, this->_fmgr_WH_dst, this->_thread_config.ptr, t2D);
}



template <>
void decx::blas::cpu_GEMM_planner<double>::Run(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst,
    decx::utils::_thread_arrange_2D* t2D)
{
    // Arrange matrix B
    decx::blas::matrix_B_arrange_64b((double*)B->Mat.ptr, (double*)this->_arranged_B._ptr.ptr, B->Pitch(), this->_arranged_B._dims.x / 4, this->_fmgr_WH_B, t2D);

    // Reshape to adapt the thread distribution of kernels
    t2D->reshape(this->GetThreadDist_dst().y, this->GetThreadDist_dst().x);

    // Execute GEMM
    decx::blas::GEMM_64b_caller<true>((double*)A->Mat.ptr, (double*)this->_arranged_B._ptr.ptr,
        (double*)dst->Mat.ptr, this->_layout_A,
        &dst->get_layout(), this->_arranged_B._dims, this->_fmgr_WH_dst, this->_thread_config.ptr, t2D, (double*)C->Mat.ptr);
}



template <bool _ABC>
void decx::blas::GEMM_fp64(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle,
    decx::_Matrix* C)
{
    if (decx::blas::g_cpu_GEMM_64b_planner._res_ptr == NULL) {
        decx::blas::g_cpu_GEMM_64b_planner.RegisterResource(new decx::blas::cpu_GEMM_planner<double>,
            5, &decx::blas::cpu_GEMM_planner<double>::Release);
    }

    decx::blas::g_cpu_GEMM_64b_planner.lock();

    // Validate the sizes of the matrices
    if constexpr (_ABC) {
        decx::blas::cpu_GEMM_planner<double>::Validate(handle, &A->get_layout(), &B->get_layout(), &C->get_layout());
    }
    else {
        decx::blas::cpu_GEMM_planner<double>::Validate(handle, &A->get_layout(), &B->get_layout());
    }
    Check_Runtime_Error(handle);

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();

    auto* _planner = decx::blas::g_cpu_GEMM_64b_planner.get_resource_raw_ptr<decx::blas::cpu_GEMM_planner<double>>();

    if (_planner->Changed(_conc, &A->get_layout(), &B->get_layout())) {
        _planner->plan(decx::cpu::_get_permitted_concurrency(), &A->get_layout(), &B->get_layout(), de::GetLastError());
        Check_Runtime_Error(handle);
    }

    decx::utils::_thread_arrange_2D t2D(_planner->GetThreadDist_B().y, _planner->GetThreadDist_B().x);
    if constexpr (_ABC) {
        _planner->Run(A, B, C, dst, &t2D);
    }
    else {
        _planner->Run(A, B, dst, &t2D);
    }

    decx::blas::g_cpu_GEMM_64b_planner.unlock();
}

template void decx::blas::GEMM_fp64<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*, decx::_Matrix*);
template void decx::blas::GEMM_fp64<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, de::DH*, decx::_Matrix*);
