/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "GEMM_utils.h"


void decx::gemm::CPUK::GEMM_fp32_cpy_L8(const float* src, float* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0, tmp_dex_dst = 0;
    for (int i = 0; i < cpy_dim.y; ++i) {
        tmp_dex_src = dex_src;
        tmp_dex_dst = dex_dst;
        for (int j = 0; j < cpy_dim.x; ++j) {
            _mm256_store_ps(dst + tmp_dex_dst, _mm256_load_ps(src + dex_src));
            tmp_dex_src += 8;
            tmp_dex_dst += 8;
        }
        dex_src += Wsrc;
        dex_dst += Wdst;
    }
}


void decx::gemm::CPUK::GEMM_fp64_cpy_L8(const double* src, double* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0, tmp_dex_dst = 0;
    for (int i = 0; i < cpy_dim.y; ++i) {
        tmp_dex_src = dex_src;
        tmp_dex_dst = dex_dst;
        for (int j = 0; j < cpy_dim.x; ++j) {
            _mm256_store_pd(dst + tmp_dex_dst, _mm256_load_pd(src + dex_src));
            tmp_dex_src += 4;
            tmp_dex_dst += 4;
        }
        dex_src += Wsrc;
        dex_dst += Wdst;
    }
}