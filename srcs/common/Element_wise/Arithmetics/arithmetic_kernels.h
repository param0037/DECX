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

#ifndef _ARITHMETIC_KERNELS_H_
#define _ARITHMETIC_KERNELS_H_

#include "../../basic.h"

#ifdef _DECX_CPU_PARTS_

#include "../../../modules/core/thread_management/thread_pool.h"
#include "../common/cpu_element_wise_planner.h"


namespace decx
{
namespace CPUK{
    _THREAD_FUNCTION_ void _add_fp32_exec(const float* A, const float* B, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _addc_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);

    _THREAD_FUNCTION_ void _sub_fp32_exec(const float* A, const float* B, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _subc_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);
    _THREAD_FUNCTION_ void _subcinv_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);

    _THREAD_FUNCTION_ void _mul_fp32_exec(const float* A, const float* B, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _mulc_fp32_exec(const float* A, float* dst, const uint64_t proc_len_v, const float constant);

    _THREAD_FUNCTION_ void _div_fp32_exec(const float* A, const float* B, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _divc_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);
    _THREAD_FUNCTION_ void _divcinv_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);

    _THREAD_FUNCTION_ void _max_fp32_exec(const float* A, const float* B, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _maxc_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);

    _THREAD_FUNCTION_ void _min_fp32_exec(const float* A, const float* B, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _minc_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v, const float constant);

    _THREAD_FUNCTION_ void _cos_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _sin_fp32_exec(const float* src, float* dst, const uint64_t proc_len_v);
}

namespace CPUK{
    _THREAD_FUNCTION_ void _add_fp64_exec(const double* A, const double* B, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _addc_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);

    _THREAD_FUNCTION_ void _sub_fp64_exec(const double* A, const double* B, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _subc_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);
    _THREAD_FUNCTION_ void _subcinv_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);

    _THREAD_FUNCTION_ void _mul_fp64_exec(const double* A, const double* B, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _mulc_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);

    _THREAD_FUNCTION_ void _div_fp64_exec(const double* A, const double* B, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _divc_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);
    _THREAD_FUNCTION_ void _divcinv_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);

    _THREAD_FUNCTION_ void _max_fp64_exec(const double* A, const double* B, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _maxc_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);

    _THREAD_FUNCTION_ void _min_fp64_exec(const double* A, const double* B, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _minc_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v, const double constant);

    _THREAD_FUNCTION_ void _cos_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v);
    _THREAD_FUNCTION_ void _sin_fp64_exec(const double* src, double* dst, const uint64_t proc_len_v);
}

namespace CPUK{
    template <typename _type_in1, typename _type_in2, typename _type_out>
    using arithmetic_kernels_1D_VVO = void(const _type_in1*, const _type_in2*, _type_out*, const uint64_t);

    template <typename _type_in, typename _type_const, typename _type_out>
    using arithmetic_kernels_1D_VCO = void(const _type_in*, _type_const*, const _type_out*, const uint64_t);

    template <typename _type_in, typename _type_out>
    using arithmetic_kernels_1D_VO = void(const _type_in*, _type_out*, const uint64_t);
}
}


#endif      // #ifdef _DECX_CPU_PARTS_

#ifdef _DECX_CUDA_PARTS_
#include "../../../modules/core/cudaStream_management/cudaStream_queue.h"
#include "../../../modules/core/cudaStream_management/cudaEvent_queue.h"

namespace decx
{

namespace GPUK{
    template <typename _type_in1, typename _type_in2, typename _type_out>
    using arithmetic_kernels_1D_VVO = void(const _type_in1*, const _type_in2*, _type_out*, const uint64_t, 
    const uint32_t, const uint32_t, decx::cuda_stream*);

    template <typename _type_in, typename _type_const, typename _type_out>
    using arithmetic_kernels_1D_VCO = void(const _type_in*, _type_const*, const _type_out*, const uint64_t, 
    const uint32_t, const uint32_t, decx::cuda_stream*);

    template <typename _type_in, typename _type_out>
    using arithmetic_kernels_1D_VO = void(const _type_in*, _type_out*, const uint64_t, const uint32_t, const uint32_t, 
    decx::cuda_stream*);
}

namespace GPUK{
    void _add_fp32_kernel(const float*, const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _addc_fp32_kernel(const float*, const float, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);

    void _sub_fp32_kernel(const float*, const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _subc_fp32_kernel(const float*, const float, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);

    void _mul_fp32_kernel(const float*, const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _mulc_fp32_kernel(const float*, const float, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);

    void _div_fp32_kernel(const float*, const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _divc_fp32_kernel(const float*, const float, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);

    void _sin_fp32_kernel(const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _cos_fp32_kernel(const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);

    void _max_fp32_kernel(const float*, const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _maxc_fp32_kernel(const float*, const float, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);

    void _min_fp32_kernel(const float*, const float*, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
    void _minc_fp32_kernel(const float*, const float, float*, const uint64_t, const uint32_t, const uint32_t, decx::cuda_stream*);
}
}

#endif

#endif
