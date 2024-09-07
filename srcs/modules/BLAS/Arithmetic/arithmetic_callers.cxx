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


#include "arithmetic_callers_LUT.h"
#include "arithmetic.h"


void decx::blas::
mat_arithmetic_caller_VVO(const decx::_Matrix*  A, 
                          const decx::_Matrix*  B, 
                          decx::_Matrix*        dst, 
                          const int32_t         arith_flag,
                          de::DH*               handle)
{
    using namespace decx::CPUK;

    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    const int32_t _kernel_dex = decx::blas::_find_arith_kernel_id<0>(arith_flag);

    void* _kernel_ptr = NULL;

    const uint64_t proc_len_flatten_v1 = static_cast<uint64_t>(A->Pitch()) * static_cast<uint64_t>(A->Height());

    switch (A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // Obtain the kernel ptr according to flag
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[0][_kernel_dex];
        // Do the plan
        _planner.plan(t1D.total_thread, proc_len_flatten_v1, sizeof(float), sizeof(float));
        // Call the kernel
        _planner.caller_binary((arithmetic_kernels_1D_VVO<float, float, float>*)_kernel_ptr,
                               (float*)A->Mat.ptr, 
                               (float*)B->Mat.ptr, 
                               (float*)dst->Mat.ptr,
                               &t1D);
        break;
    
    case de::_DATA_TYPES_FLAGS_::_FP64_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[1][_kernel_dex];

        _planner.plan(t1D.total_thread, proc_len_flatten_v1, sizeof(double), sizeof(double));

        _planner.caller_binary((arithmetic_kernels_1D_VVO<double, double, double>*)_kernel_ptr,
                               (double*)A->Mat.ptr, 
                               (double*)B->Mat.ptr, 
                               (double*)dst->Mat.ptr,
                               &t1D);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Unsupported type when performing arithmetic");
        break;
    }
}


void decx::blas::
mat_arithmetic_caller_VO(const decx::_Matrix*  src, 
                         decx::_Matrix*        dst, 
                         const int32_t         arith_flag,
                         de::DH*               handle)
{
    using namespace decx::CPUK;

    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    const int32_t _kernel_dex = decx::blas::_find_arith_kernel_id<0>(arith_flag);

    void* _kernel_ptr = NULL;

    const uint64_t proc_len_flatten_v1 = static_cast<uint64_t>(src->Pitch()) * static_cast<uint64_t>(src->Height());

    switch (src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[0][_kernel_dex];

        _planner.plan(t1D.total_thread, proc_len_flatten_v1, sizeof(float), sizeof(float));

        _planner.caller_unary((arithmetic_kernels_1D_VO<float, float>*)_kernel_ptr,
                              (float*)src->Mat.ptr, 
                              (float*)dst->Mat.ptr,
                              &t1D);
        break;
    
    case de::_DATA_TYPES_FLAGS_::_FP64_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[1][_kernel_dex];

        _planner.plan(t1D.total_thread, proc_len_flatten_v1, sizeof(double), sizeof(double));

        _planner.caller_unary((arithmetic_kernels_1D_VO<double, double>*)_kernel_ptr,
                              (double*)src->Mat.ptr, 
                              (double*)dst->Mat.ptr,
                              &t1D);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Unsupported type when performing arithmetic");
        break;
    }
}


void decx::blas::
vec_arithmetic_caller_VVO(const decx::_Vector*  A, 
                          const decx::_Vector*  B, 
                          decx::_Vector*        dst, 
                          const int32_t         arith_flag,
                          de::DH*               handle)
{
    using namespace decx::CPUK;

    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    const int32_t _kernel_dex = decx::blas::_find_arith_kernel_id<0>(arith_flag);
    void* _kernel_ptr = NULL;

    switch (A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[0][_kernel_dex];

        _planner.plan(t1D.total_thread, A->Len(), sizeof(float), sizeof(float));

        _planner.caller_binary((arithmetic_kernels_1D_VVO<float, float, float>*)_kernel_ptr,
                                (float*)A->Vec.ptr, 
                                (float*)B->Vec.ptr, 
                                (float*)dst->Vec.ptr, 
                                &t1D);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[1][_kernel_dex];

        _planner.plan(t1D.total_thread, A->Len(), sizeof(double), sizeof(double));

        _planner.caller_binary((arithmetic_kernels_1D_VVO<double, double, double>*)_kernel_ptr,
                                (double*)A->Vec.ptr, 
                                (double*)B->Vec.ptr, 
                                (double*)dst->Vec.ptr, 
                                &t1D);
        break;
    
    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Unsupported type when performing arithmetic");
        break;
    }
}



void decx::blas::
vec_arithmetic_caller_VO(const decx::_Vector*  src, 
                         decx::_Vector*        dst, 
                         const int32_t         arith_flag,
                         de::DH*               handle)
{
    using namespace decx::CPUK;

    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    const int32_t _kernel_dex = decx::blas::_find_arith_kernel_id<0>(arith_flag);
    void* _kernel_ptr = NULL;

    switch (src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[0][_kernel_dex];

        _planner.plan(t1D.total_thread, src->Len(), sizeof(float), sizeof(float));

        _planner.caller_unary((arithmetic_kernels_1D_VO<float, float>*)_kernel_ptr,
                                (float*)src->Vec.ptr, 
                                (float*)dst->Vec.ptr, 
                                &t1D);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        _kernel_ptr = g_arithmetic_cpu_kernel_LUT[1][_kernel_dex];

        _planner.plan(t1D.total_thread, src->Len(), sizeof(double), sizeof(double));

        _planner.caller_unary((arithmetic_kernels_1D_VO<double, double>*)_kernel_ptr,
                                (double*)src->Vec.ptr, 
                                (double*)dst->Vec.ptr, 
                                &t1D);
        break;
    
    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Unsupported type when performing arithmetic");
        break;
    }
}
