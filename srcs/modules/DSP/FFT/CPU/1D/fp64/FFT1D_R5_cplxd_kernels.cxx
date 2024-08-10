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
decx::dsp::fft::CPUK::_FFT1D_R5_cplxd64_1st_R2C(const double* __restrict     src, 
                                                de::CPd* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m128d recv_P0;
    __m128d recv_P1, recv_P2, recv_P3, recv_P4;
    decx::utils::simd::xmm256_reg res;
    decx::utils::simd::xmm256_reg tmp;

    const uint32_t total_Bcalc_num = signal_length / 5;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm_load_pd(src + (i << 1));
        recv_P1 = _mm_load_pd(src + ((i + total_Bcalc_num) << 1));
        recv_P2 = _mm_load_pd(src + ((i + (total_Bcalc_num << 1)) << 1));
        recv_P3 = _mm_load_pd(src + ((i + total_Bcalc_num * 3) << 1));
        recv_P4 = _mm_load_pd(src + ((i + (total_Bcalc_num << 2)) << 1));

        // Calculate the first output of the butterfly operation
        tmp._vd = _mm256_castpd128_pd256(_mm_add_pd(_mm_add_pd(_mm_add_pd(recv_P0, recv_P1),
                                          _mm_add_pd(recv_P2, recv_P3)), recv_P4));
        res._vd = _mm256_permute4x64_pd(tmp._vd, 0b11011000);
        _mm256_store_pd((double*)(dst + (i * 10)), res._vd);

        // Extend P1-P4
#if _SIMD_VER_ == AVX256
        // P0
        __m256d recv_P0_ext = _mm256_permute4x64_pd(_mm256_castpd128_pd256(recv_P0), 0b11011000);
        // P1
        __m256d recv_P1_ext = _mm256_permute2f128_pd(_mm256_castpd128_pd256(recv_P1), _mm256_castpd128_pd256(recv_P1), 0b00100000);
        recv_P1_ext = _mm256_permute4x64_pd(recv_P1_ext, 0b11011000);
        // P2
        __m256d recv_P2_ext = _mm256_permute2f128_pd(_mm256_castpd128_pd256(recv_P2), _mm256_castpd128_pd256(recv_P2), 0b00100000);
        recv_P2_ext = _mm256_permute4x64_pd(recv_P2_ext, 0b11011000);
        // P3
        __m256d recv_P3_ext = _mm256_permute2f128_pd(_mm256_castpd128_pd256(recv_P3), _mm256_castpd128_pd256(recv_P3), 0b00100000);
        recv_P3_ext = _mm256_permute4x64_pd(recv_P3_ext, 0b11011000);
        // P4
        __m256d recv_P4_ext = _mm256_permute2f128_pd(_mm256_castpd128_pd256(recv_P4), _mm256_castpd128_pd256(recv_P4), 0b00100000);
        recv_P4_ext = _mm256_permute4x64_pd(recv_P4_ext, 0b11011000);
#elif _SIMD_VER_ == AVX512
        // AVX 512 codes.

#endif

        // Calculate the second output
        res._vd = _mm256_fmadd_pd(recv_P1_ext, _mm256_setr_pd(0.309017, 0.9510565, 0.309017, 0.9510565), recv_P0_ext);
        res._vd = _mm256_fmadd_pd(recv_P2_ext, _mm256_setr_pd(-0.809017, 0.5877853, -0.809017, 0.5877853), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P3_ext, _mm256_setr_pd(-0.809017, -0.5877853, -0.809017, -0.5877853), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P4_ext, _mm256_setr_pd(0.309017, -0.9510565, 0.309017, -0.9510565), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 2), res._vd);

        // Calculate the third output
        res._vd = _mm256_fmadd_pd(recv_P1_ext, _mm256_setr_pd(-0.809017, 0.5877853, -0.809017, 0.5877853), recv_P0_ext);
        res._vd = _mm256_fmadd_pd(recv_P2_ext, _mm256_setr_pd(0.309017, -0.9510565, 0.309017, -0.9510565), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P3_ext, _mm256_setr_pd(0.309017, 0.9510565, 0.309017, 0.9510565), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P4_ext, _mm256_setr_pd(-0.809017, -0.5877853, -0.809017, -0.5877853), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 4), res._vd);

        // Calculate the fourth output
        res._vd = _mm256_fmadd_pd(recv_P1_ext,
            _mm256_setr_pd(-0.809017, -0.5877853, -0.809017, -0.5877853), recv_P0_ext);
        res._vd = _mm256_fmadd_pd(recv_P2_ext, _mm256_setr_pd(0.309017, 0.9510565, 0.309017, 0.9510565), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P3_ext, _mm256_setr_pd(0.309017, -0.9510565, 0.309017, -0.9510565), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P4_ext, _mm256_setr_pd(-0.809017, 0.5877853, -0.809017, 0.5877853), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 6), res._vd);

        // Calculate the fifth output
        res._vd = _mm256_fmadd_pd(recv_P1_ext, _mm256_setr_pd(0.309017, -0.9510565, 0.309017, -0.9510565), recv_P0_ext);
        res._vd = _mm256_fmadd_pd(recv_P2_ext, _mm256_setr_pd(-0.809017, -0.5877853, -0.809017, -0.5877853), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P3_ext, _mm256_setr_pd(-0.809017, 0.5877853, -0.809017, 0.5877853), res._vd);
        res._vd = _mm256_fmadd_pd(recv_P4_ext, _mm256_setr_pd(0.309017, 0.9510565, 0.309017, 0.9510565), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 8), res._vd);
    }
}



_THREAD_CALL_ void
decx::dsp::fft::CPUK::_FFT1D_R5_cplxd64_1st_C2C(const de::CPd* __restrict    src, 
                                                de::CPd* __restrict          dst, 
                                                const uint32_t              signal_length)
{
    __m256d recv_P0, recv_P1, recv_P2, recv_P3, recv_P4;
    decx::utils::simd::xmm256_reg res;

    const uint64_t total_Bcalc_num = signal_length / 5;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i) 
    {
        recv_P0 = _mm256_load_pd((double*)(src + (i                              << 1)));
        recv_P1 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num)          << 1)));
        recv_P2 = _mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 1))   << 1)));
        recv_P3 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num * 3)      << 1)));
        recv_P4 = _mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 2))   << 1)));

        // Calculate the first output
        res._vd = _mm256_add_pd(_mm256_add_pd(recv_P0, recv_P1), _mm256_add_pd(recv_P2, recv_P3));
        res._vd = _mm256_add_pd(recv_P4, res._vd);
        _mm256_store_pd((double*)(dst + (i * 10)), res._vd);

        // Calculate the second output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(0.309017, 0.9510565), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(-0.809017, 0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(-0.809017, -0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(0.309017, -0.9510565), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 2), res._vd);

        // Calculate the third output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(-0.809017, 0.5877853), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(0.309017, -0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(0.309017, 0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(-0.809017, -0.5877853), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 4), res._vd);

        // Calculate the fourth output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(-0.809017, -0.5877853), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(0.309017, 0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(0.309017, -0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(-0.809017, 0.5877853), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 6), res._vd);

        // Calculate the fifth output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(0.309017, -0.9510565), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(-0.809017, -0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(-0.809017, 0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(0.309017, 0.9510565), res._vd);
        _mm256_store_pd((double*)(dst + (i * 10) + 8), res._vd);
    }
}



_THREAD_FUNCTION_ void
decx::dsp::fft::CPUK::_FFT1D_R5_cplxd64_mid_C2C(const de::CPd* __restrict        src, 
                                                de::CPd* __restrict              dst, 
                                                const de::CPd* __restrict        _W_table, 
                                                const decx::dsp::fft::FKI1D*    _kernel_info)
{
    __m256d recv_P0, recv_P1, recv_P2, recv_P3, recv_P4;
    decx::utils::simd::xmm256_reg res;

    const uint64_t total_Bcalc_num = _kernel_info->_signal_len / 5;

    uint32_t dex = 0;
    uint32_t warp_loc_id;

    de::CPd W;

    const uint32_t _scale = _kernel_info->_signal_len / _kernel_info->_warp_proc_len;

    for (uint32_t i = 0; i < total_Bcalc_num; ++i)
    {
        recv_P0 = _mm256_load_pd((double*)(src + (i                              << 1)));
        recv_P1 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num)          << 1)));
        recv_P2 = _mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 1))   << 1)));
        recv_P3 = _mm256_load_pd((double*)(src + ((i + total_Bcalc_num * 3)      << 1)));
        recv_P4 = _mm256_load_pd((double*)(src + ((i + (total_Bcalc_num << 2))   << 1)));

        warp_loc_id = i % _kernel_info->_store_pitch;

        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * warp_loc_id));
        recv_P1 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P1, W);
        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * (warp_loc_id << 1)));
        recv_P2 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P2, W);
        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * warp_loc_id * 3));
        recv_P3 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P3, W);
        *((__m128d*)&W) = _mm_load_pd((double*)(_W_table + _scale * (warp_loc_id << 2)));
        recv_P4 = decx::dsp::CPUK::_cp2_mul_cp1_fp64(recv_P4, W);

        dex = (i / _kernel_info->_store_pitch) * _kernel_info->_warp_proc_len + warp_loc_id;

        // Calculate the first output
        res._vd = _mm256_add_pd(_mm256_add_pd(recv_P0, recv_P1), _mm256_add_pd(recv_P2, recv_P3));
        res._vd = _mm256_add_pd(recv_P4, res._vd);
        _mm256_store_pd((double*)(dst + (dex << 1)), res._vd);

        // Calculate the second output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(0.309017, 0.9510565), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(-0.809017, 0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(-0.809017, -0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(0.309017, -0.9510565), res._vd);
        _mm256_store_pd((double*)(dst + ((dex + _kernel_info->_store_pitch) << 1)), res._vd);

        // Calculate the third output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(-0.809017, 0.5877853), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(0.309017, -0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(0.309017, 0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(-0.809017, -0.5877853), res._vd);
        _mm256_store_pd((double*)(dst + ((dex + (_kernel_info->_store_pitch << 1)) << 1)), res._vd);

        // Calculate the fourth output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(-0.809017, -0.5877853), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(0.309017, 0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(0.309017, -0.9510565), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(-0.809017, 0.5877853), res._vd);
        _mm256_store_pd((double*)(dst + ((dex + (_kernel_info->_store_pitch * 3)) << 1)), res._vd);

        // Calculate the fifth output
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P1, de::CPd(0.309017, -0.9510565), recv_P0);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P2, de::CPd(-0.809017, -0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P3, de::CPd(-0.809017, 0.5877853), res._vd);
        res._vd = decx::dsp::CPUK::_cp2_fma_cp1_fp64(recv_P4, de::CPd(0.309017, 0.9510565), res._vd);
        _mm256_store_pd((double*)(dst + ((dex + (_kernel_info->_store_pitch << 2)) << 1)), res._vd);
    }
}
