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

#include "GEMM_callers.h"


decx::ResourceHandle decx::blas::g_cpu_GEMM_fp32_planner;

#if 1
template <bool _ABC>
void decx::blas::GEMM_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::_Matrix* C)
{
    if (decx::blas::g_cpu_GEMM_fp32_planner._res_ptr == NULL) {
        // printf("Start res reg\n");
        // auto* ptr = new decx::blas::cpu_GEMM_planner<float>;
        decx::blas::g_cpu_GEMM_fp32_planner.RegisterResource(new decx::blas::cpu_GEMM_planner<float>,
        // decx::blas::g_cpu_GEMM_fp32_planner.RegisterResource(nullptr,
            5, &decx::blas::cpu_GEMM_planner<float>::Release);
    }
    decx::blas::g_cpu_GEMM_fp32_planner.lock();

    const uint32_t _conc = DecxGetPermitConcurrency();
    
    auto* _planner = decx::blas::g_cpu_GEMM_fp32_planner.get_resource_raw_ptr<decx::blas::cpu_GEMM_planner<float>>();
    
    // Validate the sizes of the matrices
    if_opt (_ABC) {
        decx::blas::cpu_GEMM_planner<float>::Validate(&A->get_layout(), &B->get_layout(), &C->get_layout());
    }
    else {
        decx::blas::cpu_GEMM_planner<float>::Validate(&A->get_layout(), &B->get_layout());
    }
    
    // Plan if changed
    if (_planner->Changed(_conc, &A->get_layout(), &B->get_layout())) {
        _planner->plan(DecxGetPermitConcurrency(), &A->get_layout(), &B->get_layout());
    }
    
    // decx::utils::ThreadArrange2D t2D(_planner->GetThreadDist_B().y, _planner->GetThreadDist_B().x);
    if_opt (_ABC) {
        _planner->Run<false>(A, B, C, dst);
    }
    else {
        _planner->Run<false>(A, B, dst);
    }
    
    decx::blas::g_cpu_GEMM_fp32_planner.unlock();
}
#else

template <bool _ABC>
void decx::blas::GEMM_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::_Matrix* C)
{
    const uint32_t _conc = DecxGetPermitConcurrency();
    
    decx::blas::cpu_GEMM_planner<float>* _planner = new decx::blas::cpu_GEMM_planner<float>;
    
    // Validate the sizes of the matrices
    if_opt (_ABC) {
        decx::blas::cpu_GEMM_planner<float>::Validate(&A->get_layout(), &B->get_layout(), &C->get_layout());
    }
    else {
        decx::blas::cpu_GEMM_planner<float>::Validate(&A->get_layout(), &B->get_layout());
    }
    
    // Plan if changed
    if (_planner->Changed(_conc, &A->get_layout(), &B->get_layout())) {
        printf("Changed, plan again\n");
        _planner->plan(DecxGetPermitConcurrency(), &A->get_layout(), &B->get_layout());
    }
    
    // decx::utils::ThreadArrange2D t2D(_planner->GetThreadDist_B().y, _planner->GetThreadDist_B().x);
    if_opt (_ABC) {
        _planner->Run<false>(A, B, C, dst);
    }
    else {
        _planner->Run<false>(A, B, dst);
    }
}
#endif
template void decx::blas::GEMM_fp32<true>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, decx::_Matrix*);
template void decx::blas::GEMM_fp32<false>(decx::_Matrix*, decx::_Matrix*, decx::_Matrix*, decx::_Matrix*);
