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
decx::dsp::fft::CPUK::_FFT1D_R2_cplxd64_1st_R2C(const double* __restrict     src, 
                                                de::CPd* __restrict           dst, 
                                                const uint32_t               signal_length)
{
    __m128d recv_P0, recv_P1;
    decx::utils::simd::xmm256_reg res;

    const uint32_t total_Bcalc_num = (signal_length >> 1);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_pd(src + (i << 1));
        recv_P1 = _mm_load_pd(src + ((i + total_Bcalc_num) << 1));

        // Calculate the first output
        res._vd = _mm256_castpd128_pd256(_mm_add_pd(recv_P0, recv_P1));
        res._vd = _mm256_permute4x64_pd(res._vd, 0b11011000);
        _mm256_store_pd((double*)(dst + (i << 2)), res._vd);

        // Calculate the second output
        res._vd = _mm256_castpd128_pd256(_mm_sub_pd(recv_P0, recv_P1));
        res._vd = _mm256_permute4x64_pd(res._vd, 0b11011000);
        _mm256_store_pd((double*)(dst + (i << 2) + 2), res._vd);
    }
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R2_cplxd64_1st_C2C(const de::CPd* __restrict     src, 
                                                de::CPd* __restrict           dst, 
                                                const uint32_t               signal_length)
{
    __m256d recv_P0, recv_P1;
    decx::utils::simd::xmm256_reg res;

    const size_t total_Bcalc_num = (signal_length >> 1);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_load_pd((double*)(src + (i                     << 1)));
        recv_P1 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num) << 1)));

        // Calculate the first output
        res._vd = _mm256_add_pd(recv_P0, recv_P1);
        _mm256_store_pd((double*)(dst + (i << 2)), res._vd);

        // Calculate the second output
        res._vd = _mm256_sub_pd(recv_P0, recv_P1);
        _mm256_store_pd((double*)(dst + (i << 2) + 2), res._vd);
    }
}



_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R2_cplxd64_mid_C2C(const de::CPd* __restrict        src, 
                                                de::CPd* __restrict              dst, 
                                                const de::CPd* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256d recv_P0, recv_P1;
    decx::utils::simd::xmm256_reg res;

    const uint64_t total_Bcalc_num = (_kernel_info->_signal_len >> 1);

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    de::CPd W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_load_pd((double*)(src + (i                     << 1)));
        recv_P1 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num) << 1)));

        warp_loc_id = i % _kernel_info->_store_pitch;

        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * warp_loc_id));
        recv_P1 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P1, W);

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first output
        res._vd = _mm256_add_pd(recv_P0, recv_P1);
        _mm256_store_pd((double*)(dst + (dex << 1)), res._vd);

        // Calculate the second output
        res._vd = _mm256_sub_pd(recv_P0, recv_P1);
        _mm256_store_pd((double*)(dst + ((dex + _kernel_info->_store_pitch) << 1)), res._vd);
    }
}
