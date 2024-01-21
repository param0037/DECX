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
decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v4_1st(const double* __restrict        src, 
                                                        double* __restrict              dst,
                                                        const uint32_t                  _proc_len, 
                                                        const uint64_t                  _outer_dex,
                                                        const decx::dsp::fft::FIMT1D*   _Twd)
{
    decx::utils::simd::xmm256_reg reg, W_v4;
    __m128 _real_v4, _image_v4;

    const __m128 _outer_dex_v4 = _mm_setr_ps((_outer_dex) / _Twd->gap(),
                                             (_outer_dex + 1) / _Twd->gap(),
                                             (_outer_dex + 2) / _Twd->gap(),
                                             (_outer_dex + 3) / _Twd->gap());

    for (uint32_t i = 0; i < _proc_len; ++i)
    {
        reg._vd = _mm256_load_pd(src + i * 4);

        _real_v4 = _mm_mul_ps(_mm_mul_ps(_outer_dex_v4, _mm_set1_ps(i)), _mm_set1_ps(Two_Pi));

        _real_v4 = _mm_div_ps(_real_v4, _mm_set1_ps(_Twd->divide_length()));

        _image_v4 = _real_v4;
#ifdef _MSC_VER
        _real_v4 = _mm_cos_ps(_real_v4);
        _image_v4 = _mm_sin_ps(_image_v4);
#endif

        W_v4._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(_real_v4), _mm256_castps128_ps256(_image_v4), 0b00100000);
        W_v4._vf = _mm256_permutevar8x32_ps(W_v4._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        reg._vf = decx::dsp::CPUK::_cp4_mul_cp4_fp32(reg._vf, W_v4._vf);

        _mm256_store_pd(dst + i * 4, reg._vd);
    }
}




_THREAD_FUNCTION_ void 
decx::dsp::fft::CPUK::_FFT1D_Twd_smaller_kernels_v4_mid(const double* __restrict        src, 
                                                        double* __restrict              dst,
                                                        const uint32_t                  _proc_len, 
                                                        const uint64_t                  _outer_dex, 
                                                        const uint32_t                  FFT_call_time_start,
                                                        const uint32_t                  _store_pitch, 
                                                        const decx::dsp::fft::FIMT1D*   _Twd)
{
    decx::utils::simd::xmm256_reg reg, W_v4;
    __m128 _real_v4, _image_v4;

    __m128 _inner_dex_v4 = _mm_setr_ps(FFT_call_time_start + 0, FFT_call_time_start + 1, FFT_call_time_start + 2, FFT_call_time_start + 3);

    for (uint32_t i = 0; i < _proc_len; ++i) {
        reg._vd = _mm256_load_pd(src + i * 4);
        _real_v4 = _mm_mul_ps(_mm_mul_ps(_inner_dex_v4, _mm_set1_ps(_outer_dex)), _mm_set1_ps(Two_Pi));

        _real_v4 = _mm_div_ps(_real_v4, _mm_set1_ps(_Twd->divide_length()));

        _image_v4 = _real_v4;
#ifdef _MSC_VER
        _real_v4 = _mm_cos_ps(_real_v4);
        _image_v4 = _mm_sin_ps(_image_v4);
#endif

        W_v4._vf = _mm256_permute2f128_ps(_mm256_castps128_ps256(_real_v4), _mm256_castps128_ps256(_image_v4), 0b00100000);
        W_v4._vf = _mm256_permutevar8x32_ps(W_v4._vf, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        reg._vf = decx::dsp::CPUK::_cp4_mul_cp4_fp32(reg._vf, W_v4._vf);

        _mm256_store_pd(dst + i * 4, reg._vd);

        _inner_dex_v4 = _mm_add_ps(_inner_dex_v4, _mm_set1_ps(_store_pitch));
    }
}