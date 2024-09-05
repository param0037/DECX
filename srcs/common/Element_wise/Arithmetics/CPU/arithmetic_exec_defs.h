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

#ifndef _ARITHMETIC_EXEC_DEFS_H_
#define _ARITHMETIC_EXEC_DEFS_H_


#include "../../../../modules/core/thread_management/thread_pool.h"
#include "../../common/cpu_element_wise_planner.h"


/**
 * @param kernel_name Name of the kernel
 * @param type_INOUT The data type of both in and out, in arithmetic kernels, in and out have the same type
 * @param intrinsics Name of the intrinsics. In avx context, _mm(256)_xxx; In arm NEON context, vxxq_xx
*/
#define _OP_1D_VVO_(kernel_name, type_INOUT, intrinsics)        \
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
        _OP_##type_INOUT##_2I1O(intrinsics, A, B, dst);         \
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
#define _OP_1D_VCO_(kernel_name, type_INOUT, type_const, intrinsics, is_inv)    \
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
        _OP##is_inv##_##type_INOUT##_2I1O(intrinsics, src, constant, dst);      \
        _STGV_##type_INOUT(dst);                                                \
    }                                                                           \
}                                                                               \


/**
 * @param kernel_name Name of the kernel
 * @param type_INOUT The data type of both in and out, in arithmetic kernels, in and out have the same type
 * @param type_const The type of the input constant
 * @param intrinsics Name of the intrinsics. In avx context, _mm(256)_xxx; In arm NEON context, vxxq_xx
 * @param is_inv If the operation is not inversed, e.g. OP(constant, src), then left it blank; otherwise specify "inv" only
*/
#define _OP_1D_VO_(kernel_name, type_INOUT, intrinsics)       \
_THREAD_FUNCTION_ void decx::CPUK::                                   \
kernel_name(const type_INOUT* __restrict src,                         \
            type_INOUT* __restrict dst,                               \
            const uint64_t proc_len_v)                                \
{                                                                     \
    for (uint64_t i = 0; i < proc_len_v; ++i){                        \
        decx::utils::simd::xmm256_reg src_v, dst_v;                   \
        _LDGV_##type_INOUT(src);                                      \
        _OP_##type_INOUT##_1I1O(intrinsics, src, dst);                \
        _STGV_##type_INOUT(dst);                                      \
    }                                                                 \
}                                                                     \


#endif
