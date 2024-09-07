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

#ifndef _ARITHMETIC_CALLERS_LUT_H_
#define _ARITHMETIC_CALLERS_LUT_H_

#include "arithmetic.h"
#include "../../../common/Element_wise/Arithmetics/arithmetic_kernels.h"


#define _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_ 6
#define _OFFSET_OPINV_KERNEL_LUT_MAP_ 64 - _SUB_OFFSET_OPINV_KERNEL_LUT_MAP_ * 2


#ifdef _DECX_CPU_PARTS_
#include "../../../common/Element_wise/common/cpu_element_wise_planner.h"

namespace decx
{
namespace blas
{
    static void* g_arithmetic_cpu_kernel_LUT[2][18] = 
    {
        {(void*)decx::CPUK::_add_fp32_exec, 
         (void*)decx::CPUK::_mul_fp32_exec, 
         (void*)decx::CPUK::_min_fp32_exec, 
         (void*)decx::CPUK::_max_fp32_exec, 
         (void*)decx::CPUK::_sin_fp32_exec,
         (void*)decx::CPUK::_cos_fp32_exec,
         (void*)decx::CPUK::_sub_fp32_exec, 
         (void*)decx::CPUK::_div_fp32_exec, 

         (void*)decx::CPUK::_addc_fp32_exec,
         (void*)decx::CPUK::_mulc_fp32_exec,
         (void*)decx::CPUK::_minc_fp32_exec,
         (void*)decx::CPUK::_maxc_fp32_exec,
         NULL,      // Intenionally left to be blank, can't assign constant value for sin / cos
         NULL,      // Intenionally left to be blank, can't assign constant value for sin / cos
         (void*)decx::CPUK::_subc_fp32_exec,
         (void*)decx::CPUK::_divc_fp32_exec,   // 12-th
         
         (void*)decx::CPUK::_subcinv_fp32_exec,
         (void*)decx::CPUK::_divcinv_fp32_exec },

        {(void*)decx::CPUK::_add_fp64_exec, 
         (void*)decx::CPUK::_mul_fp64_exec, 
         (void*)decx::CPUK::_min_fp64_exec, 
         (void*)decx::CPUK::_max_fp64_exec, 
         (void*)decx::CPUK::_sin_fp64_exec,
         (void*)decx::CPUK::_cos_fp64_exec,
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
}
}
#endif      // #ifdef _DECX_CPU_PARTS_

namespace decx
{
namespace blas
{
    template <int32_t _is_c>
    static int32_t _find_arith_kernel_id(const int32_t _flag)
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
}
}



#ifdef _DECX_CUDA_PARTS_
#include "../../../common/Element_wise/common/cuda_element_wise_planner.h"

namespace decx
{
namespace blas
{
    static void* g_arithmetic_cuda_kernel_LUT[2][18] = 
    {
        {(void*)decx::GPUK::_add_fp32_kernel, 
         (void*)decx::GPUK::_mul_fp32_kernel, 
         (void*)decx::GPUK::_min_fp32_kernel, 
         (void*)decx::GPUK::_max_fp32_kernel, 
         (void*)decx::GPUK::_sin_fp32_kernel,
         (void*)decx::GPUK::_cos_fp32_kernel,
         (void*)decx::GPUK::_sub_fp32_kernel, 
         (void*)decx::GPUK::_div_fp32_kernel, 

         (void*)decx::GPUK::_addc_fp32_kernel,
         (void*)decx::GPUK::_mulc_fp32_kernel,
         (void*)decx::GPUK::_minc_fp32_kernel,
         (void*)decx::GPUK::_maxc_fp32_kernel,
         NULL,      // Intenionally left to be blank, can't assign constant value for sin / cos
         NULL,      // Intenionally left to be blank, can't assign constant value for sin / cos
         (void*)decx::GPUK::_subc_fp32_kernel,
         (void*)decx::GPUK::_divc_fp32_kernel,   // 12-th
         
         //(void*)decx::CPUK::_subcinv_fp32_exec,
         //(void*)decx::CPUK::_divcinv_fp32_exec },
        },
    };
}
}

#endif      // #ifdef _DECX_CUDA_PARTS_


#endif
