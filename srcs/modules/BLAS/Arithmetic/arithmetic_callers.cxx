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


#include "arithmetic.h"
#include "../../../common/element_wise/common/cpu_element_wise_planner.h"
#include "../../../common/element_wise/CPU/arithmetic_kernels.h"



#define _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_ 6
#define _OFFSET_OPINV_KERNEL_LUT_MAP_ 64 - _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_ * 2


namespace decx
{
namespace blas
{
    static void* g_arithmetic_kernel_LUT[2][18] = 
    {
        {(void*)decx::CPUK::_add_fp32_exec, 
         (void*)decx::CPUK::_mul_fp32_exec, 
         (void*)decx::CPUK::_min_fp32_exec, 
         (void*)decx::CPUK::_max_fp32_exec, 
         NULL,
         NULL,
         (void*)decx::CPUK::_sub_fp32_exec, 
         (void*)decx::CPUK::_div_fp32_exec, 

         (void*)decx::CPUK::_addc_fp32_exec,
         (void*)decx::CPUK::_mulc_fp32_exec,
         (void*)decx::CPUK::_minc_fp32_exec,
         (void*)decx::CPUK::_maxc_fp32_exec,
         NULL,
         NULL,
         (void*)decx::CPUK::_subc_fp32_exec,
         (void*)decx::CPUK::_divc_fp32_exec,   // 12-th
         
         (void*)decx::CPUK::_subcinv_fp32_exec,
         (void*)decx::CPUK::_divcinv_fp32_exec },

        {(void*)decx::CPUK::_add_fp64_exec, 
         (void*)decx::CPUK::_mul_fp64_exec, 
         (void*)decx::CPUK::_min_fp64_exec, 
         (void*)decx::CPUK::_max_fp64_exec, 
         NULL,
         NULL,
         (void*)decx::CPUK::_sub_fp64_exec, 
         (void*)decx::CPUK::_div_fp64_exec, 

         (void*)decx::CPUK::_addc_fp64_exec,
         (void*)decx::CPUK::_mulc_fp64_exec,
         (void*)decx::CPUK::_minc_fp64_exec,
         (void*)decx::CPUK::_maxc_fp64_exec,
         NULL,
         NULL,
         (void*)decx::CPUK::_subc_fp64_exec,
         (void*)decx::CPUK::_divc_fp64_exec,   // 12-th
         
         (void*)decx::CPUK::_subcinv_fp64_exec,
         (void*)decx::CPUK::_divcinv_fp64_exec },
         
    };

    template <int32_t _is_c>
    static int32_t _find_arith_kernel_id(const int32_t _flag);
}
}


template <int32_t _is_c>
static int32_t decx::blas::_find_arith_kernel_id(const int32_t _flag)
{
    if (_flag > 63){    // is _inv
        if (_flag - 64 < _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_){        // No inv-op exists, map to regular kernels
            return _flag - 64 + 32 * _is_c;
        }
        else{       // Inv-ops
            return _flag - 64 + _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_;
        }
    }
    else{       // isn't _inv
        return _flag + _is_c * _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_;
    }
}


void decx::blas::
mat_bin_arithmetic_caller(const decx::_Matrix*  A, 
                          const decx::_Matrix*  B, 
                          decx::_Matrix*        dst, 
                          const int32_t         arith_flag,
                          de::DH*               handle)
{
    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    const int32_t _kernel_dex = decx::blas::_find_arith_kernel_id<0>(arith_flag);

    void* _kernel_ptr = NULL;

    switch (A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _kernel_ptr = g_arithmetic_kernel_LUT[0][_kernel_dex];

        decx::
        arithmetic_bin_caller((decx::CPUK::arithmetic_bin_kernels<float>*)_kernel_ptr, 
                              &_planner, 
                              (float*)A->Mat.ptr, 
                              (float*)B->Mat.ptr, 
                              (float*)dst->Mat.ptr, 
                              static_cast<uint64_t>(A->Pitch()) * static_cast<uint64_t>(A->Height()), &t1D);
        break;
    
    case de::_DATA_TYPES_FLAGS_::_FP64_:
        _kernel_ptr = g_arithmetic_kernel_LUT[1][_kernel_dex];

        decx::
        arithmetic_bin_caller((decx::CPUK::arithmetic_bin_kernels<double>*)_kernel_ptr, 
                              &_planner, 
                              (double*)A->Mat.ptr, 
                              (double*)B->Mat.ptr, 
                              (double*)dst->Mat.ptr, 
                              static_cast<uint64_t>(A->Pitch()) * static_cast<uint64_t>(A->Height()), &t1D);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Unsupported type when performing arithmetic");
        break;
    }
}



void decx::blas::
vec_bin_arithmetic_caller(const decx::_Vector*  A, 
                          const decx::_Vector*  B, 
                          decx::_Vector*        dst, 
                          const int32_t         arith_flag,
                          de::DH*               handle)
{
    decx::cpu_ElementWise1D_planner _planner;
    decx::utils::_thr_1D t1D(decx::cpu::_get_permitted_concurrency());

    const int32_t _kernel_dex = decx::blas::_find_arith_kernel_id<0>(arith_flag);
    void* _kernel_ptr = NULL;

    switch (A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        _kernel_ptr = g_arithmetic_kernel_LUT[0][_kernel_dex];

        decx::
        arithmetic_bin_caller((decx::CPUK::arithmetic_bin_kernels<float>*)_kernel_ptr, 
                              &_planner, 
                              (float*)A->Vec.ptr, 
                              (float*)B->Vec.ptr, 
                              (float*)dst->Vec.ptr, 
                              A->Len(), &t1D);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        _kernel_ptr = g_arithmetic_kernel_LUT[0][_kernel_dex];

        decx::
        arithmetic_bin_caller((decx::CPUK::arithmetic_bin_kernels<double>*)_kernel_ptr, 
                              &_planner, 
                              (double*)A->Vec.ptr, 
                              (double*)B->Vec.ptr, 
                              (double*)dst->Vec.ptr, 
                              A->Len(), &t1D);
        break;
    
    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            "Unsupported type when performing arithmetic");
        break;
    }
}
