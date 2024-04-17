/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT1D_kernels.h"



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R2_cplxf32_1st_R2C(const float* __restrict     src, 
                                                double* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m128 recv_P0, recv_P1;
    decx::utils::simd::xmm256_reg res;

    const uint32_t total_Bcalc_num = (signal_length >> 1);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_ps(src + (i << 2));
        recv_P1 = _mm_load_ps(src + ((i + total_Bcalc_num) << 2));

        // Calculate the first output
        res._vf = _mm256_castps128_ps256(_mm_add_ps(recv_P0, recv_P1));
        res._vf = _mm256_permutevar8x32_ps(res._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_store_pd(dst + (i << 3), res._vd);

        // Calculate the second output
        res._vf = _mm256_castps128_ps256(_mm_sub_ps(recv_P0, recv_P1));
        res._vf = _mm256_permutevar8x32_ps(res._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_store_pd(dst + (i << 3) + 4, res._vd);
    }
}




_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R2_cplxf32_1st_C2C(const double* __restrict     src, 
                                                double* __restrict           dst, 
                                                const uint32_t               signal_length)
{
    __m256 recv_P0, recv_P1;
    decx::utils::simd::xmm256_reg res;

    const size_t total_Bcalc_num = (signal_length >> 1);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num) << 2)));

        // Calculate the first output
        res._vf = _mm256_add_ps(recv_P0, recv_P1);
        _mm256_store_pd(dst + (i << 3), res._vd);

        // Calculate the second output
        res._vf = _mm256_sub_ps(recv_P0, recv_P1);
        _mm256_store_pd(dst + (i << 3) + 4, res._vd);
    }
}




_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R2_cplxf32_mid_C2C(const double* __restrict        src, 
                                                double* __restrict              dst, 
                                                const double* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256 recv_P0, recv_P1;
    decx::utils::simd::xmm256_reg res;

    const size_t total_Bcalc_num = (_kernel_info->_signal_len >> 1);

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    de::CPf W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num) << 2)));

        warp_loc_id = i % _kernel_info->_store_pitch;

        *((double*)&W) = _W_table[_scale * warp_loc_id];
        recv_P1 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, W);

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first output
        res._vf = _mm256_add_ps(recv_P0, recv_P1);
        _mm256_store_pd(dst + (dex << 2), res._vd);

        // Calculate the second output
        res._vf = _mm256_sub_ps(recv_P0, recv_P1);
        _mm256_store_pd(dst + ((dex + _kernel_info->_store_pitch) << 2), res._vd);
    }
}
