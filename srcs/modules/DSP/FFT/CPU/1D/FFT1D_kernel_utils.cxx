/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
        
        _real_v4 = _mm_cos_ps(_real_v4);
        _image_v4 = _mm_sin_ps(_image_v4);

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


//void _mul_Twd_C2C(const de::CPd* __restrict src,
//                    de::CPd* __restrict dst,
//                    const uint32_t _signal_len,
//                    const uint32_t _prev_FFT_radix_fact_sum,
//                    const uint32_t _next_FFT_len)
//{
//    const uint32_t _divide_len = _prev_FFT_radix_fact_sum * _next_FFT_len;
//    const uint32_t _gap = _signal_len / _divide_len;
//
//    uint32_t i = 0, j = 0;
//    
//    de::CPd recv, res, W, tmp;
//
//    for (uint32_t _linear_index = 0; _linear_index < _signal_len; ++_linear_index){
//        j = (_linear_index % _prev_FFT_radix_fact_sum);
//        i = (_linear_index / _gap / _prev_FFT_radix_fact_sum);
//
//        recv = src[_linear_index];
//
//        W.construct_with_phase(Two_Pi * (double)j * i / (double)_divide_len);
//        res = decx::dsp::CPUK::_complex_mul(recv, W);
//
//        dst[_linear_index] = res;
//    }
//}


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
    _glo_idx._vi = _mm_setr_epi64x(_outer_dex * _smaller_signal_len, (_outer_dex + 1) * _smaller_signal_len);

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
#ifdef _MSC_VER
        _real_v2 = _mm_cos_pd(_real_v2);
        _image_v2 = _mm_sin_pd(_image_v2);
#endif
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
#ifdef _MSC_VER
        _real_v4 = _mm_cos_ps(_real_v4);
        _image_v4 = _mm_sin_ps(_image_v4);
#endif
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
    _glo_idx._vi = _mm_setr_epi64x(_dst_shf + (_call_times_in_warp << 1),
                                   _dst_shf + (_call_times_in_warp << 1) + 1);

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
#ifdef _MSC_VER
        _real_v2 = _mm_cos_pd(_real_v2);
        _image_v2 = _mm_sin_pd(_image_v2);
#endif
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
