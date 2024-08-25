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

#include "../arithmetic_kernels.h"
#include "../../../SIMD/intrinsics_ops.h"


#define _LDGV_float(name) name##_v._vf = _mm256_load_ps(name + i * 8)
#define _STGV_float(name) _mm256_store_ps(name + i * 8, name##_v._vf)
#define _OP_float(__intrinsics, src1, src2, dst) dst##_v._vf = __intrinsics(src1##_v._vf, src2##_v._vf)

#define _LDGV_double(name) name##_v._vd = _mm256_load_pd(name + i * 4)
#define _STGV_double(name) _mm256_store_pd(name + i * 4, name##_v._vd)
#define _OP_double(__intrinsics, src1, src2, dst) dst##_v._vd = __intrinsics(src1##_v._vd, src2##_v._vd)


#define _BINARY_1D_OP_(kernel_name, type_INOUT, intrinsics) {       \
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
}                                                                   \


_BINARY_1D_OP_(_add1D_fp32_exec, float, _mm256_add_ps)
_BINARY_1D_OP_(_sub1D_fp32_exec, float, _mm256_sub_ps)
_BINARY_1D_OP_(_mul1D_fp32_exec, float, _mm256_mul_ps)
_BINARY_1D_OP_(_div1D_fp32_exec, float, _mm256_div_ps)
_BINARY_1D_OP_(_min1D_fp32_exec, float, _mm256_min_ps)
_BINARY_1D_OP_(_max1D_fp32_exec, float, _mm256_max_ps)

_BINARY_1D_OP_(_add1D_fp64_exec, double, _mm256_add_pd)
_BINARY_1D_OP_(_sub1D_fp64_exec, double, _mm256_sub_pd)
_BINARY_1D_OP_(_mul1D_fp64_exec, double, _mm256_mul_pd)
_BINARY_1D_OP_(_div1D_fp64_exec, double, _mm256_div_pd)
_BINARY_1D_OP_(_min1D_fp64_exec, double, _mm256_min_pd)
_BINARY_1D_OP_(_max1D_fp64_exec, double, _mm256_max_pd)