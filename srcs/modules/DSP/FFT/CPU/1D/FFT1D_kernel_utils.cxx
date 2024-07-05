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


#include "FFT1D_kernel_utils.h"



_THREAD_FUNCTION_ void 
decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v4_1st(const de::CPf* __restrict        src, 
                                                        de::CPf* __restrict              dst,
                                                        const uint32_t                  _smaller_signal_len,
                                                        const uint64_t                  _outer_dex,
                                                        const decx::dsp::fft::FIMT1D*   _Twd)
{
    decx::utils::simd::xmm256_reg reg, W_v4;
    __m128 _real_v4, _image_v4;

    decx::utils::simd::xmm256_reg _glo_idx;
    _glo_idx._vi = _mm256_setr_epi64x(_outer_dex * _smaller_signal_len, 
                                     (_outer_dex + 1) * _smaller_signal_len,
                                     (_outer_dex + 2) * _smaller_signal_len,
                                     (_outer_dex + 3) * _smaller_signal_len);
    __m128 _i_v4, _j_v4;
    const uint64_t _div_len = _Twd->gap() * _Twd->_previous_fact_sum;
    
    for (uint32_t i = 0; i < _smaller_signal_len; ++i)
    {
        reg._vd = _mm256_load_pd((double*)(src + i * 4));

        _i_v4 = _mm_setr_ps(_glo_idx._arrull[0] / _div_len, _glo_idx._arrull[1] / _div_len,
                            _glo_idx._arrull[2] / _div_len, _glo_idx._arrull[3] / _div_len);
        _j_v4 = _mm_setr_ps(_glo_idx._arrull[0] % _Twd->_previous_fact_sum, _glo_idx._arrull[1] % _Twd->_previous_fact_sum,
                            _glo_idx._arrull[2] % _Twd->_previous_fact_sum, _glo_idx._arrull[3] % _Twd->_previous_fact_sum);

        _real_v4 = _mm_mul_ps(_mm_mul_ps(_i_v4, _j_v4), _mm_set1_ps(Two_Pi));

        _real_v4 = _mm_div_ps(_real_v4, _mm_set1_ps(_Twd->divide_length()));
        _image_v4 = _real_v4;

        _real_v4 = _avx_cos_fp32x4(_real_v4);
        _image_v4 = _avx_sin_fp32x4(_image_v4);

#if _SIMD_VER_ == AVX256
        W_v4._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(_real_v4), _mm256_castps128_ps256(_image_v4), 0b00100000);
        W_v4._vf = _mm256_permutevar8x32_ps(W_v4._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
#elif _SIMD_VER_ == AVX512
        // AVX512 codes.

#endif
        reg._vf = decx::dsp::CPUK::_cp4_mul_cp4_fp32(reg._vf, W_v4._vf);

        _mm256_store_pd((double*)(dst + i * 4), reg._vd);

        _glo_idx._vi = _mm256_add_epi64(_glo_idx._vi, _mm256_set1_epi64x(1));
    }
}



_THREAD_FUNCTION_ void 
decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v2_1st(const double* __restrict        src, 
                                                        double* __restrict              dst,
                                                        const uint32_t                  _smaller_signal_len, 
                                                        const uint64_t                  _outer_dex,
                                                        const decx::dsp::fft::FIMT1D*   _Twd)
{
    decx::utils::simd::xmm256_reg reg, W_v2;
    __m128d _real_v2, _image_v2;

    decx::utils::simd::xmm128_reg _glo_idx;
    //_glo_idx._vi = _mm_setr_epi64x(_outer_dex * _smaller_signal_len, (_outer_dex + 1) * _smaller_signal_len);
    _glo_idx._arrull[0] = _outer_dex * _smaller_signal_len;
    _glo_idx._arrull[1] = (_outer_dex + 1) * _smaller_signal_len;

    __m128d _i_v2, _j_v2;
    const uint64_t _div_len = _Twd->gap() * _Twd->_previous_fact_sum;
    
    for (uint32_t i = 0; i < _smaller_signal_len; ++i)
    {
        reg._vd = _mm256_load_pd(src + i * 4);

        _i_v2 = _mm_setr_pd(_glo_idx._arrull[0] / _div_len, 
                            _glo_idx._arrull[1] / _div_len);
        _j_v2 = _mm_setr_pd(_glo_idx._arrull[0] % _Twd->_previous_fact_sum,
                            _glo_idx._arrull[1] % _Twd->_previous_fact_sum);

        _real_v2 = _mm_mul_pd(_mm_mul_pd(_i_v2, _j_v2), _mm_set1_pd(Two_Pi));

        _real_v2 = _mm_div_pd(_real_v2, _mm_set1_pd(_Twd->divide_length()));

        _image_v2 = _real_v2;

        _real_v2 = _avx_cos_fp64x2(_real_v2);
        _image_v2 = _avx_sin_fp64x2(_image_v2);

#if _SIMD_VER_ == AVX256
        W_v2._vd = _mm256_permute2f128_pd(_mm256_castpd128_pd256(_real_v2), _mm256_castpd128_pd256(_image_v2), 0b00100000);
        W_v2._vd = _mm256_permute4x64_pd(W_v2._vd, 0b11011000);
#elif _SIMD_VER_ == AVX512
        // AVX512 codes.

#endif

        reg._vd = decx::dsp::CPUK::_cp2_mul_cp2_fp64(reg._vd, W_v2._vd);

        _mm256_store_pd(dst + i * 4, reg._vd);

        _glo_idx._vi = _mm_add_epi64(_glo_idx._vi, _mm_set1_epi64x(1));
    }
}



_THREAD_FUNCTION_ void 
decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v4_mid(const de::CPf* __restrict        src, 
                                                        de::CPf* __restrict              dst,
                                                        const uint32_t                  _proc_len, 
                                                        const uint32_t                  _call_times_in_warp, 
                                                        const uint64_t                  _dst_shf,
                                                        const uint32_t                  _store_pitch, 
                                                        const decx::dsp::fft::FIMT1D*   _Twd)
{
    decx::utils::simd::xmm256_reg reg, W_v4;
    __m128 _real_v4, _image_v4;

    decx::utils::simd::xmm256_reg _glo_idx;
    _glo_idx._vi = _mm256_setr_epi64x( _dst_shf + (_call_times_in_warp << 2),
                                       _dst_shf + (_call_times_in_warp << 2) + 1,
                                       _dst_shf + (_call_times_in_warp << 2) + 2,
                                       _dst_shf + (_call_times_in_warp << 2) + 3);

    __m128 _j_v4;
    const uint64_t _div_len = _Twd->gap() * _Twd->_previous_fact_sum;

    const __m128 _i_v4 = _mm_setr_ps(_glo_idx._arrull[0] / _div_len,
                                     _glo_idx._arrull[1] / _div_len,
                                     _glo_idx._arrull[2] / _div_len,
                                     _glo_idx._arrull[3] / _div_len);

    for (uint32_t i = 0; i < _proc_len; ++i)
    {
        reg._vd = _mm256_load_pd((double*)(src + i * 2));

        _j_v4 = _mm_setr_ps(_glo_idx._arrull[0] % _Twd->_previous_fact_sum,
                            _glo_idx._arrull[1] % _Twd->_previous_fact_sum,
                            _glo_idx._arrull[2] % _Twd->_previous_fact_sum,
                            _glo_idx._arrull[3] % _Twd->_previous_fact_sum);

        _real_v4 = _mm_mul_ps(_mm_mul_ps(_i_v4, _j_v4), _mm_set1_ps(Two_Pi));

        _real_v4 = _mm_div_ps(_real_v4, _mm_set1_ps(_Twd->divide_length()));

        _image_v4 = _real_v4;

        _real_v4 = _avx_cos_fp32x4(_real_v4);
        _image_v4 = _avx_sin_fp32x4(_image_v4);

#if _SIMD_VER_ == AVX256
        W_v4._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(_real_v4), _mm256_castps128_ps256(_image_v4), 0b00100000);
        W_v4._vf = _mm256_permutevar8x32_ps(W_v4._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
#elif _SIMD_VER_ == AVX512
        // AVX512 codes.

#endif
        reg._vf = decx::dsp::CPUK::_cp4_mul_cp4_fp32(reg._vf, W_v4._vf);

        _mm256_store_pd((double*)(dst + i * 2), reg._vd);

        _glo_idx._vi = _mm256_add_epi64(_glo_idx._vi, _mm256_set1_epi64x(_store_pitch));
    }
}


_THREAD_FUNCTION_ void 
decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v2_mid(const de::CPd* __restrict        src, 
                                                        de::CPd* __restrict              dst,
                                                        const uint32_t                  _proc_len, 
                                                        const uint32_t                  _call_times_in_warp, 
                                                        const uint64_t                  _dst_shf,
                                                        const uint32_t                  _store_pitch, 
                                                        const decx::dsp::fft::FIMT1D*   _Twd)
{
    decx::utils::simd::xmm256_reg reg, W_v2;
    __m128d _real_v2, _image_v2;

    decx::utils::simd::xmm128_reg _glo_idx;
//     _glo_idx._vi = _mm_setr_epi64x(_dst_shf + (_call_times_in_warp << 1),
//                                    _dst_shf + (_call_times_in_warp << 1) + 1);

    _glo_idx._arrull[0] = _dst_shf + (_call_times_in_warp << 1);
    _glo_idx._arrull[1] = _dst_shf + (_call_times_in_warp << 1) + 1;

    __m128d _j_v2;
    const uint64_t _div_len = _Twd->gap() * _Twd->_previous_fact_sum;
    const __m128d _i_v2 = _mm_setr_pd(_glo_idx._arrull[0] / _div_len,
                                      _glo_idx._arrull[1] / _div_len);

    for (uint32_t i = 0; i < _proc_len; ++i)
    {
        reg._vd = _mm256_load_pd((double*)(src + i * 2));

        _j_v2 = _mm_setr_pd(_glo_idx._arrull[0] % _Twd->_previous_fact_sum,
                            _glo_idx._arrull[1] % _Twd->_previous_fact_sum);

        _real_v2 = _mm_mul_pd(_mm_mul_pd(_i_v2, _j_v2), _mm_set1_pd(Two_Pi));

        _real_v2 = _mm_div_pd(_real_v2, _mm_set1_pd(_Twd->divide_length()));

        _image_v2 = _real_v2;

        _real_v2 = _avx_cos_fp64x2(_real_v2);
        _image_v2 = _avx_sin_fp64x2(_image_v2);

#if _SIMD_VER_ == AVX256
        W_v2._vd = _mm256_permute2f128_pd(_mm256_castpd128_pd256(_real_v2), _mm256_castpd128_pd256(_image_v2), 0b00100000);
        W_v2._vd = _mm256_permute4x64_pd(W_v2._vd, 0b11011000);
#elif _SIMD_VER_ == AVX512
        // AVX512 codes.

#endif

        reg._vd = decx::dsp::CPUK::_cp2_mul_cp2_fp64(reg._vd, W_v2._vd);

        _mm256_store_pd((double*)(dst + i * 2), reg._vd);

        _glo_idx._vi = _mm_add_epi64(_glo_idx._vi, _mm_set1_epi64x(_store_pitch));
    }
}
