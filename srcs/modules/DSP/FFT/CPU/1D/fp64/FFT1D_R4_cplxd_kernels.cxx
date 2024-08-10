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
decx::dsp::fft::CPUK::_FFT1D_R4_cplxd64_1st_R2C(const double* __restrict     src, 
                                                de::CPd* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m128d recv_P0, recv_P1, recv_P2, recv_P3;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp;

    const uint32_t total_Bcalc_num = (signal_length >> 2);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_pd(src + (i                              << 1));
        recv_P1 = _mm_load_pd(src + ((i + total_Bcalc_num)          << 1));
        recv_P2 = _mm_load_pd(src + ((i + (total_Bcalc_num << 1))   << 1));
        recv_P3 = _mm_load_pd(src + ((i + total_Bcalc_num * 3)      << 1));

        // Calculate the first output of the butterfly operation
        tmp._vd = _mm256_castpd128_pd256(_mm_add_pd(_mm_add_pd(recv_P0, recv_P1), _mm_add_pd(recv_P2, recv_P3)));
        res._vd = _mm256_permute4x64_pd(tmp._vd, 0b11011000);
        // Store the first output of the butterfly operation
        _mm256_store_pd((double*)(dst + (i << 3)), res._vd);

        // Calculate the second output
        // [R0, R1, I0, I1]
#if _SIMD_VER_ == AVX256
        tmp._vd = _mm256_permute2f128_pd(_mm256_castpd128_pd256(_mm_sub_pd(recv_P0, recv_P2)), 
                                         _mm256_castpd128_pd256(_mm_sub_pd(recv_P1, recv_P3)), 0x20);
        res._vd = _mm256_permute4x64_pd(tmp._vd, 0b11011000);
#elif _SIMD_VER_ == AVX512
        // AVX512 codes.
#endif
        // Store the second output
        _mm256_store_pd((double*)(dst + (i << 3) + 2), res._vd);
        // Store the fourth output
        // Reverse the signs of the imaginary parts of the four parallel complexes
        res._vi = _mm256_xor_si256(res._vi, 
                                   _mm256_setr_epi64x(0, 0x8000000000000000, 0, 0x8000000000000000));
        _mm256_store_pd((double*)(dst + (i << 3) + 6), res._vd);

        // Calculate the third output
        tmp._vd = _mm256_castpd128_pd256(_mm_add_pd(_mm_sub_pd(recv_P0, recv_P1), _mm_sub_pd(recv_P2, recv_P3)));
        res._vd = _mm256_permute4x64_pd(tmp._vd, 0b11011000);
        // Store the third output of the butterfly operation
        _mm256_store_pd((double*)(dst + (i << 3) + 4), res._vd);
    }
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R4_cplxd64_1st_C2C(const de::CPd* __restrict     src, 
                                                de::CPd* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m256d recv_P0, recv_P1, recv_P2, recv_P3;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp1, tmp2;

    const size_t total_Bcalc_num = (signal_length >> 2);

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_load_pd((double*)(src + (i                             << 1)));
        recv_P1 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num)         << 1)));
        recv_P2 = _mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 1))  << 1)));
        recv_P3 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num * 3)     << 1)));

        // Calculate the first and third output
        tmp1._vd = _mm256_add_pd(recv_P0, recv_P2);
        tmp2._vd = _mm256_add_pd(recv_P1, recv_P3);
        // Store the first output
        res._vd = _mm256_add_pd(tmp1._vd, tmp2._vd);
        _mm256_store_pd((double*)(dst + (i << 3)), res._vd);
        // Store the third output
        res._vd = _mm256_sub_pd(tmp1._vd, tmp2._vd);
        _mm256_store_pd((double*)(dst + (i << 3) + 4), res._vd);

        // Calculate the second and the fourth output
        tmp1._vd = _mm256_permute_pd(_mm256_sub_pd(recv_P1, recv_P3), 0b0101);
        tmp1._vi = _mm256_xor_si256(tmp1._vi, _mm256_setr_epi64x(0x8000000000000000, 0, 0x8000000000000000, 0));

        tmp2._vd = _mm256_sub_pd(recv_P0, recv_P2);

        res._vd = _mm256_add_pd(tmp2._vd, tmp1._vd);
        // Store the second output
        _mm256_store_pd((double*)(dst + (i << 3) + 2), res._vd);

        tmp1._vi = _mm256_xor_si256(tmp1._vi, _mm256_set1_epi64x(0x8000000000000000));
        res._vd = _mm256_add_pd(tmp2._vd, tmp1._vd);
        // Store the fourth output
        _mm256_store_pd((double*)(dst + (i << 3) + 6), res._vd);
    }
}



_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R4_cplxd64_mid_C2C(const de::CPd* __restrict        src, 
                                                de::CPd* __restrict              dst, 
                                                const de::CPd* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256d recv_P0, recv_P1, recv_P2, recv_P3;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp1, tmp2;

    const uint64_t total_Bcalc_num = (_kernel_info->_signal_len >> 2);

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    de::CPd W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_load_pd((double*)(src + (i                             << 1)));
        recv_P1 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num)         << 1)));
        recv_P2 = _mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 1))  << 1)));
        recv_P3 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num * 3)     << 1)));

        warp_loc_id = i % _kernel_info->_store_pitch;

        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * warp_loc_id));
        recv_P1 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P1, W);
        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * warp_loc_id * 2));
        recv_P2 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P2, W);
        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * warp_loc_id * 3));
        recv_P3 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P3, W);

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first and third output
        tmp1._vd = _mm256_add_pd(recv_P0, recv_P2);
        tmp2._vd = _mm256_add_pd(recv_P1, recv_P3);
        // Store the first output
        res._vd = _mm256_add_pd(tmp1._vd, tmp2._vd);
        _mm256_store_pd((double*)(dst + (dex << 1)), res._vd);
        // Store the third output
        res._vd = _mm256_sub_pd(tmp1._vd, tmp2._vd);
        _mm256_store_pd((double*)(dst + ((dex + (_kernel_info->_store_pitch << 1)) << 1)), res._vd);

        // Calculate the second and the fourth output
        tmp1._vd = _mm256_permute_pd(_mm256_sub_pd(recv_P1, recv_P3), 0b0101);
        tmp1._vi = _mm256_xor_si256(tmp1._vi, _mm256_setr_epi64x(0x8000000000000000, 0, 0x8000000000000000, 0));

        tmp2._vd = _mm256_sub_pd(recv_P0, recv_P2);

        res._vd = _mm256_add_pd(tmp2._vd, tmp1._vd);
        _mm256_store_pd((double*)(dst + ((dex + _kernel_info->_store_pitch) << 1)), res._vd);

        tmp1._vi = _mm256_xor_si256(tmp1._vi, _mm256_set1_epi64x(0x8000000000000000));
        res._vd = _mm256_add_pd(tmp2._vd, tmp1._vd);
        // Store the fourth output
        _mm256_store_pd((double*)(dst + ((dex + _kernel_info->_store_pitch * 3) << 1)), res._vd);
    }
}
