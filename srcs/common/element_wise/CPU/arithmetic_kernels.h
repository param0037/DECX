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
#include "../../../modules/core/thread_management/thread_pool.h"
#include "../common/cpu_element_wise_planner.h"


/**
 * @param kernel_name Name of the kernel
 * @param type_INOUT The data type of both in and out, in arithmetic kernels, in and out have the same type
 * @param intrinsics Name of the intrinsics. In avx context, _mm(256)_xxx; In arm NEON context, vxxq_xx
*/
#define _BINARY_1D_OP_(kernel_name, type_INOUT, intrinsics)        \
    _THREAD_FUNCTION_ void decx::CPUK::                             \
    kernel_name(const type_INOUT* __restrict A,                     \
                const type_INOUT* __restrict B,                     \
                type_INOUT* __restrict dst,                         \
                const uint64_t proc_len_v)                          \
    {                                                               \
        for (uint64_t i = 0; i < proc_len_v; ++i){                  \
            decx::utils::simd::xmm256_reg A_v, B_v, dst_v;          \
            _LDGV_##type_INOUT(A);                                  \
            _LDGV_##type_INOUT(B);                                  \
            _OP_##type_INOUT(intrinsics, A, B, dst);                \
            _STGV_##type_INOUT(dst);                                \
        }                                                           \
    }                                                               \


/**
 * @param kernel_name Name of the kernel
 * @param type_INOUT The data type of both in and out, in arithmetic kernels, in and out have the same type
 * @param type_const The type of the input constant
 * @param intrinsics Name of the intrinsics. In avx context, _mm(256)_xxx; In arm NEON context, vxxq_xx
 * @param is_inv If the operation is not inversed, e.g. OP(constant, src), then left it blank; otherwise specify "inv" only
*/
#define _UNARY_1D_OP_(kernel_name, type_INOUT, type_const, intrinsics, is_inv)     \
    _THREAD_FUNCTION_ void decx::CPUK::                                             \
    kernel_name(const type_INOUT* __restrict src,                                   \
                type_INOUT* __restrict dst,                                         \
                const uint64_t proc_len_v,                                          \
                const type_const constant)                                          \
    {                                                                               \
        for (uint64_t i = 0; i < proc_len_v; ++i){                                  \
            decx::utils::simd::xmm256_reg src_v, constant_v, dst_v;                 \
            _LDGV_##type_INOUT(src);                                                \
            _DUPV_##type_INOUT(constant);                                           \
            _OP##is_inv##_##type_INOUT(intrinsics, src, constant, dst);               \
            _STGV_##type_INOUT(dst);                                                \
        }                                                                           \
    }                                                                               \



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
}


namespace CPUK{
    template <typename _type_inout>
    using arithmetic_bin_kernels = void(const _type_inout*, const _type_inout*, _type_inout*, const uint64_t);

    template <typename _type_inout>
    using arithmetic_un_kernels = void(const _type_inout*, _type_inout*, const uint64_t, const _type_inout);
}


template <typename _type_inout>
static void arithmetic_bin_caller(decx::CPUK::arithmetic_bin_kernels<_type_inout>* _kernel, 
    decx::cpu_ElementWise1D_planner* _planner, const _type_inout* A, const _type_inout* B, _type_inout* dst,
    const uint64_t _proc_len, decx::utils::_thr_1D *t1D);


template <typename _type_inout>
static void arithmetic_un_caller(decx::CPUK::arithmetic_bin_kernels<_type_inout>* _kernel, 
    decx::cpu_ElementWise1D_planner* _planner, const _type_inout* src, const _type_inout constant, _type_inout* dst,
    const uint64_t _proc_len, decx::utils::_thr_1D *t1D);
}


template <typename _type_inout> static void decx::
arithmetic_bin_caller(decx::CPUK::arithmetic_bin_kernels<_type_inout>* _kernel, 
                      decx::cpu_ElementWise1D_planner* _planner, 
                      const _type_inout* A, 
                      const _type_inout* B, 
                      _type_inout* dst,
                      const uint64_t _proc_len,
                      decx::utils::_thr_1D *t1D)
{
    _planner->plan(t1D->total_thread, _proc_len, sizeof(_type_inout), sizeof(_type_inout));

    _planner->caller_binary(_kernel, A, B, dst, t1D);
}



template <typename _type_inout> static void decx::
arithmetic_un_caller(decx::CPUK::arithmetic_bin_kernels<_type_inout>* _kernel, 
                      decx::cpu_ElementWise1D_planner* _planner, 
                      const _type_inout* src, 
                      const _type_inout constant, 
                      _type_inout* dst,
                      const uint64_t _proc_len,
                      decx::utils::_thr_1D *t1D)
{
    _planner->plan(t1D->total_thread, _proc_len, sizeof(_type_inout), sizeof(_type_inout));

    _planner->caller_unary(_kernel, src, dst, t1D, constant);
}



#endif