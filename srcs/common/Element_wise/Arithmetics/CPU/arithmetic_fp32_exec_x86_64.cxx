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
#define _DUPV_float(name) name##_v._vf = _mm256_set1_ps(name)
#define _OP_float(__intrinsics, src1, src2, dst) dst##_v._vf = __intrinsics(src1##_v._vf, src2##_v._vf)
#define _OPinv_float(__intrinsics, src1, src2, dst) dst##_v._vf = __intrinsics(src2##_v._vf, src1##_v._vf)


// Binary
_BINARY_1D_OP_(_add_fp32_exec, float, _mm256_add_ps)
_BINARY_1D_OP_(_sub_fp32_exec, float, _mm256_sub_ps)
_BINARY_1D_OP_(_mul_fp32_exec, float, _mm256_mul_ps)
_BINARY_1D_OP_(_div_fp32_exec, float, _mm256_div_ps)
_BINARY_1D_OP_(_min_fp32_exec, float, _mm256_min_ps)
_BINARY_1D_OP_(_max_fp32_exec, float, _mm256_max_ps)

// Unary (binary with constant)
_UNARY_1D_OP_(_addc_fp32_exec,      float, float, _mm256_add_ps, )
_UNARY_1D_OP_(_subc_fp32_exec,      float, float, _mm256_sub_ps, )
_UNARY_1D_OP_(_subcinv_fp32_exec,   float, float, _mm256_sub_ps, inv)
_UNARY_1D_OP_(_mulc_fp32_exec,      float, float, _mm256_mul_ps, )
_UNARY_1D_OP_(_divc_fp32_exec,      float, float, _mm256_div_ps, )
_UNARY_1D_OP_(_divcinv_fp32_exec,   float, float, _mm256_div_ps, inv)
_UNARY_1D_OP_(_minc_fp32_exec,      float, float, _mm256_min_ps, )
_UNARY_1D_OP_(_maxc_fp32_exec,      float, float, _mm256_max_ps, )