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
decx::dsp::fft::CPUK::_FFT1D_R5_cplxf32_1st_R2C(const float* __restrict     src, 
                                                double* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m128 recv_P0;
    __m128 recv_P1, recv_P2, recv_P3, recv_P4;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp;

    const uint32_t total_Bcalc_num = signal_length / 5;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_ps(src + (i << 2));
        recv_P1 = _mm_load_ps(src + ((i + total_Bcalc_num) << 2));
        recv_P2 = _mm_load_ps(src + ((i + (total_Bcalc_num << 1)) << 2));
        recv_P3 = _mm_load_ps(src + ((i + total_Bcalc_num * 3) << 2));
        recv_P4 = _mm_load_ps(src + ((i + (total_Bcalc_num << 2)) << 2));

        // Calculate the first output of the butterfly operation
        tmp._vf = _mm256_castps128_ps256(_mm_add_ps(_mm_add_ps(_mm_add_ps(recv_P0, recv_P1),
                                          _mm_add_ps(recv_P2, recv_P3)), recv_P4));
        res._vf = _mm256_permutevar8x32_ps(tmp._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_store_pd(dst + (i * 20), res._vd);

        // Extend P1-P4
        // P0
        __m256 recv_P0_ext = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(recv_P0), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // P1
        __m256 recv_P1_ext = _mm256_permute2f128_ps(_mm256_castps128_ps256(recv_P1), _mm256_castps128_ps256(recv_P1), 0b00100000);
        recv_P1_ext = _mm256_permutevar8x32_ps(recv_P1_ext, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // P2
        __m256 recv_P2_ext = _mm256_permute2f128_ps(_mm256_castps128_ps256(recv_P2), _mm256_castps128_ps256(recv_P2), 0b00100000);
        recv_P2_ext = _mm256_permutevar8x32_ps(recv_P2_ext, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // P3
        __m256 recv_P3_ext = _mm256_permute2f128_ps(_mm256_castps128_ps256(recv_P3), _mm256_castps128_ps256(recv_P3), 0b00100000);
        recv_P3_ext = _mm256_permutevar8x32_ps(recv_P3_ext, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // P4
        __m256 recv_P4_ext = _mm256_permute2f128_ps(_mm256_castps128_ps256(recv_P4), _mm256_castps128_ps256(recv_P4), 0b00100000);
        recv_P4_ext = _mm256_permutevar8x32_ps(recv_P4_ext, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        // Calculate the second output
        res._vf = _mm256_fmadd_ps(recv_P1_ext,
            _mm256_setr_ps(0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565),
            recv_P0_ext);
        res._vf = _mm256_fmadd_ps(recv_P2_ext,
            _mm256_setr_ps(-0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P3_ext,
            _mm256_setr_ps(-0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P4_ext,
            _mm256_setr_ps(0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565),
            res._vf);
        _mm256_store_pd(dst + (i * 20) + 4, res._vd);

        // Calculate the third output
        res._vf = _mm256_fmadd_ps(recv_P1_ext,
            _mm256_setr_ps(-0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853),
            recv_P0_ext);
        res._vf = _mm256_fmadd_ps(recv_P2_ext,
            _mm256_setr_ps(0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P3_ext,
            _mm256_setr_ps(0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P4_ext,
            _mm256_setr_ps(-0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853),
            res._vf);
        _mm256_store_pd(dst + (i * 20) + 8, res._vd);

        // Calculate the fourth output
        res._vf = _mm256_fmadd_ps(recv_P1_ext,
            _mm256_setr_ps(-0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853),
            recv_P0_ext);
        res._vf = _mm256_fmadd_ps(recv_P2_ext,
            _mm256_setr_ps(0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P3_ext,
            _mm256_setr_ps(0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P4_ext,
            _mm256_setr_ps(-0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853),
            res._vf);
        _mm256_store_pd(dst + (i * 20) + 12, res._vd);

        // Calculate the fifth output
        res._vf = _mm256_fmadd_ps(recv_P1_ext,
            _mm256_setr_ps(0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565, 0.309017, -0.9510565),
            recv_P0_ext);
        res._vf = _mm256_fmadd_ps(recv_P2_ext,
            _mm256_setr_ps(-0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853, -0.809017, -0.5877853),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P3_ext,
            _mm256_setr_ps(-0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853, -0.809017, 0.5877853),
            res._vf);
        res._vf = _mm256_fmadd_ps(recv_P4_ext,
            _mm256_setr_ps(0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565, 0.309017, 0.9510565),
            res._vf);
        _mm256_store_pd(dst + (i * 20) + 16, res._vd);
    }
}




_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R5_cplxf32_1st_C2C(const double* __restrict    src, 
                                                double* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m256 recv_P0, recv_P1, recv_P2, recv_P3, recv_P4;
    decx::utils::simd::xmm256_reg res;

    const size_t total_Bcalc_num = signal_length / 5;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd(src + (i                              << 2)));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num)          << 2)));
        recv_P2 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + (total_Bcalc_num << 1))   << 2)));
        recv_P3 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num * 3)      << 2)));
        recv_P4 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + (total_Bcalc_num << 2))   << 2)));

        // Calculate the first output
        res._vf = _mm256_add_ps(_mm256_add_ps(recv_P0, recv_P1), _mm256_add_ps(recv_P2, recv_P3));
        res._vf = _mm256_add_ps(recv_P4, res._vf);
        _mm256_store_pd(dst + (i * 20), res._vd);

        // Calculate the second output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(0.309017, 0.9510565), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(0.309017, -0.9510565), res._vf);
        _mm256_store_pd(dst + (i * 20) + 4, res._vd);

        // Calculate the third output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(-0.809017, 0.5877853), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(-0.809017, -0.5877853), res._vf);
        _mm256_store_pd(dst + (i * 20) + 8, res._vd);

        // Calculate the fourth output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(-0.809017, -0.5877853), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(-0.809017, 0.5877853), res._vf);
        _mm256_store_pd(dst + (i * 20) + 12, res._vd);

        // Calculate the fifth output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(0.309017, -0.9510565), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(0.309017, 0.9510565), res._vf);
        _mm256_store_pd(dst + (i * 20) + 16, res._vd);
    }
}





_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R5_cplxf32_mid_C2C(const double* __restrict        src, 
                                                double* __restrict              dst, 
                                                const double* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256 recv_P0, recv_P1, recv_P2, recv_P3, recv_P4;
    decx::utils::simd::xmm256_reg res;

    const size_t total_Bcalc_num = _kernel_info->_signal_len / 5;

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    de::CPf W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd(src + (i                              << 2)));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num)          << 2)));
        recv_P2 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + (total_Bcalc_num << 1))   << 2)));
        recv_P3 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num * 3)      << 2)));
        recv_P4 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + (total_Bcalc_num << 2))   << 2)));

        warp_loc_id = i % _kernel_info->_store_pitch;

        *((double*)&W) = _W_table[_scale * warp_loc_id];
        recv_P1 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, W);
        *((double*)&W) = _W_table[_scale * (warp_loc_id << 1)];
        recv_P2 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, W);
        *((double*)&W) = _W_table[_scale * warp_loc_id * 3];
        recv_P3 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P3, W);
        *((double*)&W) = _W_table[_scale * (warp_loc_id << 2)];
        recv_P4 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P4, W);

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first output
        res._vf = _mm256_add_ps(_mm256_add_ps(recv_P0, recv_P1), _mm256_add_ps(recv_P2, recv_P3));
        res._vf = _mm256_add_ps(recv_P4, res._vf);
        _mm256_store_pd(dst + (dex << 2), res._vd);

        // Calculate the second output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(0.309017, 0.9510565), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(0.309017, -0.9510565), res._vf);
        _mm256_store_pd(dst + ((dex + _kernel_info->_store_pitch) << 2), res._vd);

        // Calculate the third output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(-0.809017, 0.5877853), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(-0.809017, -0.5877853), res._vf);
        _mm256_store_pd(dst + ((dex + (_kernel_info->_store_pitch << 1)) << 2), res._vd);

        // Calculate the fourth output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(-0.809017, -0.5877853), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(-0.809017, 0.5877853), res._vf);
        _mm256_store_pd(dst + ((dex + (_kernel_info->_store_pitch * 3)) << 2), res._vd);

        // Calculate the fifth output
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P1, de::CPf(0.309017, -0.9510565), recv_P0);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P2, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P3, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::CPUK::_cp4_fma_cp1_fp32(recv_P4, de::CPf(0.309017, 0.9510565), res._vf);
        _mm256_store_pd(dst + ((dex + (_kernel_info->_store_pitch << 2)) << 2), res._vd);
    }
}