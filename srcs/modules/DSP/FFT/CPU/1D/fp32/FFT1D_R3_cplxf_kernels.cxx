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


#include "../FFT1D_kernels.h"



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R3_cplxf32_1st_R2C(const float* __restrict     src, 
                                                de::CPf* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m128 recv_P0, recv_P1, recv_P2, tmp1, tmp2;
    decx::utils::simd::xmm256_reg res;

    const uint32_t total_Bcalc_num = signal_length / 3;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_ps(src + (i << 2));
        recv_P1 = _mm_load_ps(src + ((i + total_Bcalc_num) << 2));
        recv_P2 = _mm_load_ps(src + ((i + (total_Bcalc_num << 1)) << 2));

        tmp1 = _mm_add_ps(recv_P1, recv_P2);
        tmp2 = _mm_mul_ps(_mm_sub_ps(recv_P1, recv_P2), _mm_set1_ps(0.8660254f));

        // Calculate the first output
        res._vf = _mm256_castps128_ps256(_mm_add_ps(recv_P0, tmp1));
        res._vf = _mm256_permutevar8x32_ps(res._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_store_pd((double*)(dst + (i * 12)), res._vd);

        // Calculate the second output
        tmp1 = _mm_fmadd_ps(tmp1, _mm_set1_ps(-0.5f), recv_P0);
        res._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(tmp1), _mm256_castps128_ps256(tmp2), 0b00100000);
        res._vf = _mm256_permutevar8x32_ps(res._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_store_pd((double*)(dst + (i * 12) + 4), res._vd);

        // Calculate the third output
        res._vi = _mm256_xor_si256(res._vi,
            _mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
        _mm256_store_pd((double*)(dst + (i * 12) + 8), res._vd);
    }
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R3_cplxf32_1st_C2C(const de::CPf* __restrict     src, 
                                                de::CPf* __restrict           dst, 
                                                const uint32_t               signal_length)
{
    __m256 recv_P0, recv_P1, recv_P2;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp1, tmp2;

    const size_t total_Bcalc_num = signal_length / 3;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd((double*)(src + (i << 2))));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd((double*)(src + ((i + total_Bcalc_num) << 2))));
        recv_P2 = _mm256_castpd_ps(_mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 1)) << 2))));

        // Calculate the first output
        res._vf = _mm256_add_ps(_mm256_add_ps(recv_P0, recv_P1), recv_P2);
        _mm256_store_pd((double*)(dst + (i * 12)), res._vd);

        // Calculate the second output
        res._vf = _mm256_add_ps(decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, de::CPf(-0.5, 0.8660254f)), recv_P0);
        res._vf = _mm256_add_ps(res._vf, decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, de::CPf(-0.5, -0.8660254f)));
        _mm256_store_pd((double*)(dst + (i * 12) + 4), res._vd);

        // Calculate the third output
        res._vf = _mm256_add_ps(decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, de::CPf(-0.5, -0.8660254f)), recv_P0);
        res._vf = _mm256_add_ps(res._vf, decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, de::CPf(-0.5, 0.8660254f)));
        _mm256_store_pd((double*)(dst + (i * 12) + 8), res._vd);
    }
}




_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R3_cplxf32_mid_C2C(const de::CPf* __restrict        src, 
                                                de::CPf* __restrict              dst, 
                                                const de::CPf* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256 recv_P0, recv_P1, recv_P2;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp1, tmp2;

    const size_t total_Bcalc_num = _kernel_info->_signal_len / 3;

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    de::CPf W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_castpd_ps(_mm256_load_pd((double*)(src + (i << 2))));
        recv_P1 = _mm256_castpd_ps(_mm256_load_pd((double*)(src + ((i + total_Bcalc_num) << 2))));
        recv_P2 = _mm256_castpd_ps(_mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 1)) << 2))));

        warp_loc_id = i % _kernel_info->_store_pitch;

        W = _W_table[_scale * warp_loc_id];
        recv_P1 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, W);
        W = _W_table[_scale * warp_loc_id * 2];
        recv_P2 = decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, W);

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first output
        res._vf = _mm256_add_ps(_mm256_add_ps(recv_P0, recv_P1), recv_P2);
        _mm256_store_pd((double*)(dst + (dex << 2)), res._vd);

        // Calculate the second output
        res._vf = _mm256_add_ps(decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, de::CPf(-0.5, 0.8660254f)), recv_P0);
        res._vf = _mm256_add_ps(res._vf, decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, de::CPf(-0.5, -0.8660254f)));
        _mm256_store_pd((double*)(dst + ((dex + _kernel_info->_store_pitch) << 2)), res._vd);

        // Calculate the third output
        res._vf = _mm256_add_ps(decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P1, de::CPf(-0.5, -0.8660254f)), recv_P0);
        res._vf = _mm256_add_ps(res._vf, decx::dsp::CPUK::_cp4_mul_cp1_fp32(recv_P2, de::CPf(-0.5, 0.8660254f)));
        _mm256_store_pd((double*)(dst + ((dex + (_kernel_info->_store_pitch << 1)) << 2)), res._vd);
    }
}
