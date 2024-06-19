/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "matrix_B_arrange.h"
#include "GEMM_callers.h"


namespace de
{
    namespace blas {
        namespace cpu {
            _DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst);

            _DECX_API_ void GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);
        }
    }
}



_DECX_API_ void de::blas::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix& dst)
{
    de::ResetLastError();

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::blas::g_cpu_GEMM_fp32_planner._res_ptr == NULL) {
        decx::blas::g_cpu_GEMM_fp32_planner.RegisterResource(new decx::blas::cpu_GEMM_planner<float>,
            5, &decx::blas::cpu_GEMM_planner<float>::Release);
    }

    decx::blas::g_cpu_GEMM_fp32_planner.lock();

    auto* _planner = decx::blas::g_cpu_GEMM_fp32_planner.get_resource_raw_ptr<decx::blas::cpu_GEMM_planner<float>>();
    _planner->plan(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout(), de::GetLastError());
    if (de::GetLastError()->error_type != decx::DECX_error_types::DECX_SUCCESS) {
        return;
    }
    decx::utils::_thread_arrange_2D t2D(_planner->GetThreadDist_B().y, _planner->GetThreadDist_B().x);

    _planner->Run(_A, _B, _dst, &t2D);

    decx::blas::g_cpu_GEMM_fp32_planner.unlock();
}



_DECX_API_ void de::blas::cpu::GEMM(de::Matrix& A, de::Matrix& B, de::Matrix &C, de::Matrix& dst)
{
    de::ResetLastError();

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Matrix* _C = dynamic_cast<decx::_Matrix*>(&C);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (decx::blas::g_cpu_GEMM_fp32_planner._res_ptr == NULL) {
        decx::blas::g_cpu_GEMM_fp32_planner.RegisterResource(new decx::blas::cpu_GEMM_planner<float>,
            5, &decx::blas::cpu_GEMM_planner<float>::Release);
    }

    decx::blas::g_cpu_GEMM_fp32_planner.lock();

    auto* _planner = decx::blas::g_cpu_GEMM_fp32_planner.get_resource_raw_ptr<decx::blas::cpu_GEMM_planner<float>>();
    _planner->plan(decx::cpu::_get_permitted_concurrency(), &_A->get_layout(), &_B->get_layout(), de::GetLastError());
    if (de::GetLastError()->error_type != decx::DECX_error_types::DECX_SUCCESS) {
        return;
    }
    decx::utils::_thread_arrange_2D t2D(_planner->GetThreadDist_B().y, _planner->GetThreadDist_B().x);

    _planner->Run(_A, _B, _C, _dst, &t2D);

    decx::blas::g_cpu_GEMM_fp32_planner.unlock();
}
