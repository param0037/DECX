/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _RCP2_SLIDING_WINDOW_MACROS_H_
#define _RCP2_SLIDING_WINDOW_MACROS_H_


#include "../../../core/basic.h"


#define _rcp2_flex_fp32_(_pramed_func) {                \
    __m256 res_vec8;                                    \
    size_t dex_src = 0, dex_dst = 0;                    \
                                                        \
    for (int i = 0; i < proc_dim.y; ++i) {              \
        for (int j = 0; j < proc_dim.x; ++j) {          \
            res_vec8 = _pramed_func;                    \
            _mm256_store_ps(dst + dex_dst, res_vec8);   \
            dex_src += 8;                               \
            dex_dst += 8;                               \
        }                                               \
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);     \
        dex_src += (Wsrc << 3) - (proc_dim.x << 3);     \
    }                                                   \
}


#define _rcp2_flex_fp32_CCOEFF(_pramed_func) {                \
    __m256 res_vec8;                                    \
    size_t dex_src = 0, dex_dst = 0, dex_Isrc = 0;                    \
                                                        \
    for (int i = 0; i < proc_dim.y; ++i) {              \
        for (int j = 0; j < proc_dim.x; ++j) {          \
            res_vec8 = _pramed_func;                    \
            _mm256_store_ps(dst + dex_dst, res_vec8);   \
            dex_src += 8;                               \
            dex_dst += 8;                               \
            dex_Isrc += 8;                               \
        }                                               \
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);     \
        dex_src += (Wsrc << 3) - (proc_dim.x << 3);     \
        dex_Isrc += (W_Isrc << 3) - (proc_dim.x << 3);     \
    }                                                   \
}



#define _rcp2_fixed_fp32_(_pramed_func) {                       \
    __m256 res_vec8;                                            \
    size_t dex_src = 0, dex_dst = 0;                            \
                                                                \
    for (int i = 0; i < _BLOCKED_RCP2_FP32_H_; ++i) {           \
        for (int j = 0; j < _BLOCKED_RCP2_FP32_W_; ++j) {       \
            res_vec8 = _pramed_func;                            \
            _mm256_store_ps(dst + dex_dst, res_vec8);           \
            dex_src += 8;                                       \
            dex_dst += 8;                                       \
        }                                                       \
        dex_dst += (Wdst << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  \
        dex_src += (Wsrc << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  \
    }                                                           \
}


#define _rcp2_fixed_fp32_CCOEFF(_pramed_func) {                       \
    __m256 res_vec8;                                            \
    size_t dex_src = 0, dex_dst = 0, dex_Isrc = 0;                            \
                                                                \
    for (int i = 0; i < _BLOCKED_RCP2_FP32_H_; ++i) {           \
        for (int j = 0; j < _BLOCKED_RCP2_FP32_W_; ++j) {       \
            res_vec8 = _pramed_func;                            \
            _mm256_store_ps(dst + dex_dst, res_vec8);           \
            dex_src += 8;                                       \
            dex_dst += 8;                                       \
            dex_Isrc += 8;                                       \
        }                                                       \
        dex_dst += (Wdst << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  \
        dex_src += (Wsrc << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  \
        dex_Isrc += (W_Isrc << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  \
    }                                                           \
}



#endif