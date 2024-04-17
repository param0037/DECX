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
decx::dsp::fft::CPUK::_FFT1D_R4_cplxf32_1st_R2C(const float* __restrict     src, 
                                                double* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m128 recv_P0, recv_P1, recv_P2, recv_P3;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp;

    const uint32_t total_Bcalc_num = (signal_length >> 2);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_ps(src + (i << 2));
        recv_P1 = _mm_load_ps(src + ((i + total_Bcalc_num) << 2));
        recv_P2 = _mm_load_ps(src + ((i + (total_Bcalc_num << 1)) << 2));
        recv_P3 = _mm_load_ps(src + ((i + total_Bcalc_num * 3) << 2));

        // Calculate the first output of the butterfly operation
        tmp._vf = _mm256_castps128_ps256(_mm_add_ps(_mm_add_ps(recv_P0, recv_P1), _mm_add_ps(recv_P2, recv_P3)));
        res._vf = _mm256_permutevar8x32_ps(tmp._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // Store the first output of the butterfly operation
        _mm256_store_pd(dst + (i << 4), res._vd);

        // Calculate the second output
        // [R0, R1, R2, R3, I0, I1, I2, I3]
        tmp._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(_mm_sub_ps(recv_P0, recv_P2)), 
                                         _mm256_castps128_ps256(_mm_sub_ps(recv_P1, recv_P3)), 0x20);
        res._vf = _mm256_permutevar8x32_ps(tmp._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // Store the second output
        _mm256_store_pd(dst + (i << 4) + 4, res._vd);
        // Store the fourth output
        // Reverse the signs of the imaginary parts of the four parallel complexes
        res._vi = _mm256_xor_si256(res._vi, 
                                   _mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
        _mm256_store_pd(dst + (i << 4) + 12, res._vd);

        // Calculate the third output
        tmp._vf = _mm256_castps128_ps256(_mm_add_ps(_mm_sub_ps(recv_P0, recv_P1), _mm_sub_ps(recv_P2, recv_P3)));
        res._vf = _mm256_permutevar8x32_ps(tmp._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        // Store the third output of the butterfly operation
        _mm256_store_pd(dst + (i << 4) + 8, res._vd);
    }
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R4_cplxf32_1st_C2C(const double* __restrict     src, 
                                                double* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m256 recv_P0, recv_P1, recv_P2, recv_P3;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp1, tmp2;

    const size_t total_Bcalc_num = (signal_length >> 2);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num) << 2)));
        recv_P2 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + (total_Bcalc_num << 1)) << 2)));
        recv_P3 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num * 3) << 2)));

        // Calculate the first and third output
        tmp1._vf = _mm256_add_ps(recv_P0, recv_P2);
        tmp2._vf = _mm256_add_ps(recv_P1, recv_P3);
        // Store the first output
        res._vf = _mm256_add_ps(tmp1._vf, tmp2._vf);
        _mm256_store_pd(dst + (i << 4), res._vd);
        // Store the third output
        res._vf = _mm256_sub_ps(tmp1._vf, tmp2._vf);
        _mm256_store_pd(dst + (i << 4) + 8, res._vd);

        // Calculate the second and the fourth output
        tmp1._vi = _mm256_xor_si256(_mm256_castps_si256(_mm256_permute_ps(recv_P1, 0b10110001)),
                                    _mm256_setr_epi32(0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0));
        res._vi = _mm256_xor_si256(_mm256_castps_si256(_mm256_permute_ps(recv_P3, 0b10110001)),
                                   _mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
        tmp2._vf = _mm256_add_ps(tmp1._vf, res._vf);

        res._vf = _mm256_sub_ps(recv_P0, recv_P2);
        tmp1._vf = _mm256_add_ps(res._vf, tmp2._vf);
        // Store the second output
        _mm256_store_pd(dst + (i << 4) + 4, tmp1._vd);
        tmp1._vf = _mm256_sub_ps(res._vf, tmp2._vf);
        // Store the fourth output
        _mm256_store_pd(dst + (i << 4) + 12, tmp1._vd);
    }
}



_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R4_cplxf32_mid_C2C(const double* __restrict        src, 
                                                double* __restrict              dst, 
                                                const double* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256 recv_P0, recv_P1, recv_P2, recv_P3;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp1, tmp2;

    const size_t total_Bcalc_num = (_kernel_info->_signal_len >> 2);

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    decx::utils::simd::xmm256_reg W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd(src + (i << 2)));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num) << 2)));
        recv_P2 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + (total_Bcalc_num << 1)) << 2)));
        recv_P3 = _mm256_castpd_ps(_mm256_load_pd(src + ((i + total_Bcalc_num * 3) << 2)));

        warp_loc_id = i % _kernel_info->_store_pitch;

        W._vd = _mm256_i32gather_pd(_W_table, _mm_setr_epi32(0, 
                                                             _scale * warp_loc_id, 
                                                             _scale * warp_loc_id * 2, 
                                                             _scale * warp_loc_id * 3), 8);
        
        recv_P1 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, *((de::CPf*)&W._arrd[1]));
        recv_P2 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, *((de::CPf*)&W._arrd[2]));
        recv_P3 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P3, *((de::CPf*)&W._arrd[3]));

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first and third output
        tmp1._vf = _mm256_add_ps(recv_P0, recv_P2);
        tmp2._vf = _mm256_add_ps(recv_P1, recv_P3);
        // Store the first output
        res._vf = _mm256_add_ps(tmp1._vf, tmp2._vf);
        _mm256_store_pd(dst + (dex << 2), res._vd);
        // Store the third output
        res._vf = _mm256_sub_ps(tmp1._vf, tmp2._vf);
        _mm256_store_pd(dst + ((dex + (_kernel_info->_store_pitch << 1)) << 2), res._vd);

        // Calculate the second and the fourth output
        tmp1._vi = _mm256_xor_si256(_mm256_castps_si256(_mm256_permute_ps(recv_P1, 0b10110001)),
                                    _mm256_setr_epi32(0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0));
        res._vi = _mm256_xor_si256(_mm256_castps_si256(_mm256_permute_ps(recv_P3, 0b10110001)),
                                   _mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
        tmp2._vf = _mm256_add_ps(tmp1._vf, res._vf);

        res._vf = _mm256_sub_ps(recv_P0, recv_P2);
        tmp1._vf = _mm256_add_ps(res._vf, tmp2._vf);
        // Store the second output
        _mm256_store_pd(dst + ((dex + _kernel_info->_store_pitch) << 2), tmp1._vd);
        tmp1._vf = _mm256_sub_ps(res._vf, tmp2._vf);
        // Store the fourth output
        _mm256_store_pd(dst + ((dex + _kernel_info->_store_pitch * 3) << 2), tmp1._vd);
    }
}