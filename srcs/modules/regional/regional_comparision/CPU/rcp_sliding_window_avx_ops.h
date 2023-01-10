/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _RCP_SLIDING_WINDOW_AVX_OPS_H_
#define _RCP_SLIDING_WINDOW_AVX_OPS_H_

#include "../../../core/basic.h"


namespace decx
{
    namespace conv {
        struct __align__(64) _packed2_v256_int32 {
            __m256i _v1, _v2;
        };

        struct __align__(64) _packed2_v256_fp32 {
            __m256 _v1, _v2;
        };

        typedef _packed2_v256_int32 _v256_2i32;
        typedef _packed2_v256_fp32 _v256_2f32;

        typedef union __align__(64) _packed2_v256 {
            _v256_2i32 _vi32;
            _v256_2f32 _vf32;
        }_v256;
    }
}


#define _BLOCKED_RCP2_FP32_H_ 8
#define _BLOCKED_RCP2_FP32_W_ 8
#define _BLOCKED_RCP2_UINT8_H_ 8
#define _BLOCKED_RCP2_UINT8_W_ 8


#define _SLIDING_WINDOW_FP32_GENERAL_(dex, _moving, _static, tmp) {                                                     \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                      \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permutevar8x32_ps(_static, _mm256_set1_epi32(dex)), 0b00100001);    \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                                \
}


#define _SLIDING_WINDOW_FP32_LOAD_0_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(0, 3, 2, 1)), 0b00100001);         \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}



#define _SLIDING_WINDOW_FP32_LOAD_1_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(1, 2, 3, 0)), 0b00100001);         \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}


#define _SLIDING_WINDOW_FP32_LOAD_2_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(0, 3, 2, 1)), 0b00100001);         \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}


#define _SLIDING_WINDOW_FP32_LOAD_3_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _static, 0b00100001);                                                     \
    _moving = _mm256_blend_ps(_proc_reg, tmp, 0b10001000);                                                          \
}


#define _SLIDING_WINDOW_FP32_LOAD_4_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(0, 3, 2, 1)), 0b00110001);         \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}


#define _SLIDING_WINDOW_FP32_LOAD_5_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(1, 2, 3, 0)), 0b00110001);         \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}


#define _SLIDING_WINDOW_FP32_LOAD_6_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _mm256_permute_ps(_static, _MM_SHUFFLE(2, 3, 1, 0)), 0b00110001);         \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}



#define _SLIDING_WINDOW_FP32_LOAD_7_(_moving, _static, tmp) {                                                       \
    _moving = _mm256_permute_ps(_moving, _MM_SHUFFLE(0, 3, 2, 1));                                                  \
    tmp = _mm256_permute2f128_ps(_moving, _static, 0b00110001);                                                     \
    _moving = _mm256_blend_ps(_moving, tmp, 0b10001000);                                                            \
}


// -------------------------------------------------------------------------------------------------------------------


#define _SLIDING_WINDOW_FP64_SHIFT_LOAD_3_(_moving, _static) {         \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                     \
    _moving = _mm256_blend_pd(_moving, _static, 0b1000);                    \
}


#define _SLIDING_WINDOW_FP64_SHIFT_LOAD_2_(_moving, _static) {                         \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                                     \
    _moving = _mm256_blend_pd(_moving, _mm256_permute_pd(_static, 0b0101), 0b1000);         \
}


#define _SLIDING_WINDOW_FP64_SHIFT_LOAD_0_(_moving, _static) {                                                     \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                                                                 \
    _moving = _mm256_blend_pd(_moving, _mm256_permute_pd(_mm256_permute2f128_pd(_static, _static, 1), 0b0101), 0b1000); \
}


#define _SLIDING_WINDOW_FP64_SHIFT_LOAD_1_(_moving, _static) {                                 \
    _moving = decx::utils::simd::_mm256d_shift1_H2L(_moving);                                             \
    _moving = _mm256_blend_pd(_moving, _mm256_permute2f128_pd(_static, _static, 1), 0b1000);        \
}


// ----------------------------------------- uint8 ------------------------------------

#define _SLIDING_WINDOW_UINT8_SHIFT_(_moving_reg, _shf) {    \
    _moving_reg = _mm_castpd_si128(_mm_loadu_pd((double*)(_store_reg + (_shf))));  /* ? */     \
}


#endif