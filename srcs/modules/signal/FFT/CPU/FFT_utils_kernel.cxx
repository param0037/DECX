/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT_utils_kernel.h"


_THREAD_FUNCTION_ void
decx::signal::CPUK::_FFT1D_cpy_cvtcp_f32(const double* __restrict src, float* __restrict dst, const size_t proc_len)
{
    size_t dex = 0;
    __m256d recv;
    __m128 res;
    for (int i = 0; i < proc_len; ++i) {
        recv = _mm256_load_pd(src + dex);
        res = _mm256_castps256_ps128(
            _mm256_permutevar8x32_ps(_mm256_castpd_ps(recv), _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
        _mm_store_ps(dst + dex, res);
        dex += 4;
    }
}


_THREAD_FUNCTION_ void
decx::signal::CPUK::_FFT2D_transpose_C(const double* __restrict src, 
                                 double* __restrict dst,
                                 const uint Wsrc,           // Width of src, in element
                                 const uint Wdst,           // Width of dst, in element
                                 const uint2 proc_dim,
                                 const uint _L4)      // ~.x -> width; ~.y -> height of dst (BOTH IN VEC4)
{
    //_MM_TRANSPOSE4_PS
    __m256d row[4];
    size_t dex_src, tmp_dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dim.y; ++i) 
    {
        dex_src = tmp_dex_src;
        for (int j = 0; j < proc_dim.x; ++j) {
            row[0] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
            row[1] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
            row[2] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
            row[3] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;

            // transpose
            __m256d tmp0, tmp1, tmp2, tmp3;
            tmp0 = _mm256_shuffle_pd(row[0], row[1], 0x0);
            tmp2 = _mm256_shuffle_pd(row[0], row[1], 0xF);
            tmp1 = _mm256_shuffle_pd(row[2], row[3], 0x0);
            tmp3 = _mm256_shuffle_pd(row[2], row[3], 0xF);

            row[0] = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
            row[1] = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
            row[2] = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
            row[3] = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

            _mm256_store_pd(dst + dex_dst, row[0]);
            _mm256_store_pd(dst + dex_dst + Wdst, row[1]);
            _mm256_store_pd(dst + dex_dst + Wdst * 2, row[2]);
            _mm256_store_pd(dst + dex_dst + Wdst * 3, row[3]);
            dex_dst += 4;
        }
        tmp_dex_src += 4;
        dex_dst += 3 * Wdst;
    }

    dex_src = tmp_dex_src;
    for (int j = 0; j < proc_dim.x; ++j) {
        row[0] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
        row[1] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
        row[2] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
        row[3] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;

        // transpose
        __m256d tmp0, tmp1, tmp2, tmp3;
        tmp0 = _mm256_shuffle_pd(row[0], row[1], 0x0);
        tmp2 = _mm256_shuffle_pd(row[0], row[1], 0xF);
        tmp1 = _mm256_shuffle_pd(row[2], row[3], 0x0);
        tmp3 = _mm256_shuffle_pd(row[2], row[3], 0xF);

        row[0] = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
        row[1] = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
        row[2] = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
        row[3] = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

        for (int k = 0; k < _L4; ++k) {
            _mm256_store_pd(dst + dex_dst + Wdst * k, row[k]);
        }
        dex_dst += 4;
    }
}



_THREAD_FUNCTION_ void
decx::signal::CPUK::_FFT2D_transpose_C_and_divide(const double* __restrict src, 
                                            double* __restrict dst,
                                            const uint Wsrc,           // Width of src, in element
                                            const uint Wdst,           // Width of dst, in element
                                            const uint2 proc_dim,
                                            const float _denominator,
                                            const uint _L4)      // ~.x -> width; ~.y -> height of dst (BOTH IN VEC4)
{
    //_MM_TRANSPOSE4_PS
    __m256d row[4];
    size_t dex_src, tmp_dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dim.y; ++i) 
    {
        dex_src = tmp_dex_src;
        for (int j = 0; j < proc_dim.x; ++j) {
            row[0] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
            row[1] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
            row[2] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
            row[3] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;

            row[0] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[0]), _mm256_set1_ps(_denominator)));
            row[1] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[1]), _mm256_set1_ps(_denominator)));
            row[2] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[2]), _mm256_set1_ps(_denominator)));
            row[3] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[3]), _mm256_set1_ps(_denominator)));

            // transpose
            __m256d tmp0, tmp1, tmp2, tmp3;
            tmp0 = _mm256_shuffle_pd(row[0], row[1], 0x0);
            tmp2 = _mm256_shuffle_pd(row[0], row[1], 0xF);
            tmp1 = _mm256_shuffle_pd(row[2], row[3], 0x0);
            tmp3 = _mm256_shuffle_pd(row[2], row[3], 0xF);

            row[0] = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
            row[1] = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
            row[2] = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
            row[3] = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

            _mm256_store_pd(dst + dex_dst, row[0]);
            _mm256_store_pd(dst + dex_dst + Wdst, row[1]);
            _mm256_store_pd(dst + dex_dst + Wdst * 2, row[2]);
            _mm256_store_pd(dst + dex_dst + Wdst * 3, row[3]);
            dex_dst += 4;
        }
        tmp_dex_src += 4;
        dex_dst += 3 * Wdst;
    }

    dex_src = tmp_dex_src;
    for (int j = 0; j < proc_dim.x; ++j) {
        row[0] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
        row[1] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
        row[2] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;
        row[3] = _mm256_load_pd(src + dex_src);   dex_src += Wsrc;

        row[0] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[0]), _mm256_set1_ps(_denominator)));
        row[1] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[1]), _mm256_set1_ps(_denominator)));
        row[2] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[2]), _mm256_set1_ps(_denominator)));
        row[3] = _mm256_castps_pd(_mm256_div_ps(_mm256_castpd_ps(row[3]), _mm256_set1_ps(_denominator)));

        // transpose
        __m256d tmp0, tmp1, tmp2, tmp3;
        tmp0 = _mm256_shuffle_pd(row[0], row[1], 0x0);
        tmp2 = _mm256_shuffle_pd(row[0], row[1], 0xF);
        tmp1 = _mm256_shuffle_pd(row[2], row[3], 0x0);
        tmp3 = _mm256_shuffle_pd(row[2], row[3], 0xF);

        row[0] = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
        row[1] = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
        row[2] = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
        row[3] = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

        for (int k = 0; k < _L4; ++k) {
            _mm256_store_pd(dst + dex_dst + Wdst * k, row[k]);
        }
        dex_dst += 4;
    }
}



_THREAD_FUNCTION_ void
decx::signal::CPUK::_FFT2D_transpose_C2R_and_divide(const double* __restrict src, 
                                            float* __restrict dst,
                                            const uint Wsrc,           // Width of src, in element
                                            const uint Wdst,           // Width of dst, in element
                                            const uint2 proc_dim,
                                            const float _denominator,
                                            const uint _L4)      // ~.x -> width; ~.y -> height of dst (BOTH IN VEC4)
{
    //_MM_TRANSPOSE4_PS
    __m256 I_buffer;
    __m128 row[4];
    size_t dex_src, tmp_dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dim.y; ++i) 
    {
        dex_src = tmp_dex_src;
        for (int j = 0; j < proc_dim.x; ++j) {
            I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
            row[0] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
            dex_src += Wsrc;
            I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
            row[1] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
            dex_src += Wsrc;
            I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
            row[2] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
            dex_src += Wsrc;
            I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
            row[3] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
            dex_src += Wsrc;

            row[0] = _mm_div_ps(row[0], _mm_set1_ps(_denominator));
            row[1] = _mm_div_ps(row[1], _mm_set1_ps(_denominator));
            row[2] = _mm_div_ps(row[2], _mm_set1_ps(_denominator));
            row[3] = _mm_div_ps(row[3], _mm_set1_ps(_denominator));

            // transpose
            _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);

            _mm_store_ps(dst + dex_dst, row[0]);
            _mm_store_ps(dst + dex_dst + Wdst, row[1]);
            _mm_store_ps(dst + dex_dst + Wdst * 2, row[2]);
            _mm_store_ps(dst + dex_dst + Wdst * 3, row[3]);
            dex_dst += 4;
        }
        tmp_dex_src += 4;
        dex_dst += 3 * Wdst;
    }

    dex_src = tmp_dex_src;
    for (int j = 0; j < proc_dim.x; ++j) {
        I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
        row[0] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
        dex_src += Wsrc;
        I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
        row[1] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
        dex_src += Wsrc;
        I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
        row[2] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
        dex_src += Wsrc;
        I_buffer = _mm256_castpd_ps(_mm256_load_pd(src + dex_src));
        row[3] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(I_buffer, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
        dex_src += Wsrc;

        row[0] = _mm_div_ps(row[0], _mm_set1_ps(_denominator));
        row[1] = _mm_div_ps(row[1], _mm_set1_ps(_denominator));
        row[2] = _mm_div_ps(row[2], _mm_set1_ps(_denominator));
        row[3] = _mm_div_ps(row[3], _mm_set1_ps(_denominator));

        // transpose
        _MM_TRANSPOSE4_PS(row[0], row[1], row[2], row[3]);

        for (int k = 0; k < _L4; ++k) {
            _mm_store_ps(dst + dex_dst + Wdst * k, row[k]);
        }
        dex_dst += 4;
    }
}