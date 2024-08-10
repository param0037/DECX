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

#ifndef _SHF_MM256_FP32_H_
#define _SHF_MM256_FP32_H_

/**
 * Given two __m256 regs, moving and static. Assume that the elements are: [x0,~x7]
 * and [x8,~x15]. The moving reg will be changed after calling these macros.
*/

#include "../../basic.h"


#define _SHF_MM256_FP32_GENERAL_(dex, _moving, _static, tmp) {                                                     \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                 \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permutevar8x32_ps(_static, _mm256_set1_epi32(dex)), 0b00100001);  \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                           \
}

/**
 * _moving = [x1,~x8]
*/
#define _SHF_MM256_FP32_SHF_1_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(0, 3, 2, 1)), 0b00100001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}

/**
 * _moving = [x2,~x9]
*/
#define _SHF_MM256_FP32_SHF_2_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(1, 2, 3, 0)), 0b00100001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}

/**
 * _moving = [x3,~x10]
*/
#define _SHF_MM256_FP32_SHF_3_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(2, 3, 1, 0)), 0b00100001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}

/**
 * _moving = [x4,~x11]
*/
#define _SHF_MM256_FP32_SHF_4_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _static, 0b00100001);                                                \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}

/**
 * _moving = [x5,~x12]
*/
#define _SHF_MM256_FP32_SHF_5_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(0, 3, 2, 1)), 0b00110001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}

/**
 * _moving = [x6,~x13]
*/
#define _SHF_MM256_FP32_SHF_6_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(1, 2, 3, 0)), 0b00110001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}

/**
 * _moving = [x7,~x14]
*/
#define _SHF_MM256_FP32_SHF_7_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                             \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(2, 3, 1, 0)), 0b00110001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                       \
}


// ---------------------------------------------------------------------------------------------------------------


#define _SHF_MM256_FP64_SHF_3_(_moving, _static) {                            \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                       \
    _moving = _mm256_blend_pd(_moving, _mm256_permute_pd(_static, 0b0101), 0b1000); \
}


#define _SHF_MM256_FP64_SHF_1_(_moving, _static) {                                                                \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                                                           \
    _moving = _mm256_blend_pd(_moving, _mm256_permute_pd(_mm256_permute2f128_pd(_static, _static, 1), 0b0101), 0b1000); \
}


#define _SHF_MM256_FP64_SHF_2_(_moving, _static) {                                        \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                                   \
    _moving = _mm256_blend_pd(_moving, _mm256_permute2f128_pd(_static, _static, 1), 0b1000);    \
}


// ----------------------------------------- uint8 ------------------------------------

#define _SHF_MM256_UINT8_SHIFT_(_moving, _static, _shf) {               \
    _moving >>= 8;                                                      \
    _moving |= ((_static << ((8 - _shf) * 8)) & 0xFF00000000000000);    \
}


#endif
