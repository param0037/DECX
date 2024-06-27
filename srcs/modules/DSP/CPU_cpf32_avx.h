/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef CPU_CPF32_AVX_H_
#define CPU_CPF32_AVX_H_

#include "../core/basic.h"
#include "../core/thread_management/thread_pool.h"
#include "../classes/classes_util.h"


#define _expand_CPf32_MM256_(__real, __image) _mm256_setr_ps(__real, __image,       \
                                                             __real, __image,       \
                                                             __real, __image,       \
                                                             __real, __image)       \



namespace decx
{
namespace dsp
{
namespace CPUK{
    inline _THREAD_CALL_
    __m256 _cp4_mul_cp4_fp32(const __m256 __x, const __m256 __y)
    {
        __m256 rr_ii = _mm256_mul_ps(__x, __y);
        __m256 ri_ir = _mm256_mul_ps(__x, _mm256_permute_ps(__y, 0b10110001));
        // real, real | image, image
        rr_ii = _mm256_permute_ps(rr_ii, 0b11011000);
        ri_ir = _mm256_permute_ps(ri_ir, 0b11011000);
        __m256 res = _mm256_unpacklo_ps(rr_ii, ri_ir);
        res = _mm256_addsub_ps(res, _mm256_unpackhi_ps(rr_ii, ri_ir));
        return res;
    }


    inline _THREAD_CALL_
    __m256 _cp4_mul_cp1_fp32(const __m256 __x, const de::CPf __y)
    {
        __m256 filled_y = _mm256_castpd_ps(_mm256_set1_pd(*((double*)&__y)));
        __m256 rr_ii = _mm256_mul_ps(__x, filled_y);
        __m256 ri_ir = _mm256_mul_ps(__x, _mm256_permute_ps(filled_y, 0b10110001));
        // real, real | image, image
        rr_ii = _mm256_permute_ps(rr_ii, 0b11011000);
        ri_ir = _mm256_permute_ps(ri_ir, 0b11011000);
        __m256 res = _mm256_unpacklo_ps(rr_ii, ri_ir);
        res = _mm256_addsub_ps(res, _mm256_unpackhi_ps(rr_ii, ri_ir));
        return res;
    }


    inline _THREAD_CALL_
    __m256 _cp4_conjugate_fp32(const __m256 __x)
    {
        return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(__x),
            _mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000)));
    }


    inline _THREAD_CALL_
    /*
    * @return __x * __y + __z
    */
    __m256 _cp4_fma_cp4_fp32(const __m256 __x, const __m256 __y, const __m256 __z)
    {
        __m256 rr_ii = _mm256_mul_ps(__x, __y);
        __m256 ri_ir = _mm256_mul_ps(__x, _mm256_permute_ps(__y, 0b10110001));
        // real, real | image, image
        rr_ii = _mm256_permute_ps(rr_ii, 0b11011000);
        ri_ir = _mm256_permute_ps(ri_ir, 0b11011000);
        __m256 res = _mm256_unpacklo_ps(rr_ii, ri_ir);
        res = _mm256_addsub_ps(res, _mm256_unpackhi_ps(rr_ii, ri_ir));
        res = _mm256_add_ps(res, __z);
        return res;
    }


    inline _THREAD_CALL_
    /*
    * @return __x * __y + __z
    */
    __m256 _cp4_fma_cp1_fp32(const __m256 __x, const de::CPf __y, const __m256 __z)
    {
        const __m256 __y_v4 = _mm256_castpd_ps(_mm256_set1_pd(*((double*)&__y)));
        __m256 rr_ii = _mm256_mul_ps(__x, __y_v4);
        __m256 ri_ir = _mm256_mul_ps(__x, _mm256_permute_ps(__y_v4, 0b10110001));
        // real, real | image, image
        rr_ii = _mm256_permute_ps(rr_ii, 0b11011000);
        ri_ir = _mm256_permute_ps(ri_ir, 0b11011000);
        __m256 res = _mm256_unpacklo_ps(rr_ii, ri_ir);
        res = _mm256_addsub_ps(res, _mm256_unpackhi_ps(rr_ii, ri_ir));
        res = _mm256_add_ps(res, __z);
        return res;
    }


    static _THREAD_CALL_ de::CPf _complex_mul(const de::CPf a, const de::CPf b)
    {
        de::CPf res;
        //res.real = (a.real * b.real - a.image * b.image);
        res.real = fma(a.real, b.real, -1.f * a.image * b.image);
        //res.image = (a.real * b.image + a.image * b.real);
        res.image = fma(a.real, b.image, a.image * b.real);
        return res;
    }


    inline _THREAD_CALL_ de::CPf _complex_add(const de::CPf a, const de::CPf b)
    {
        de::CPf res;
        res.real = a.real + b.real;
        res.image = a.image + b.image;
        return res;
    }


    inline _THREAD_CALL_ de::CPf _complex_sub(const de::CPf a, const de::CPf b)
    {
        de::CPf res;
        res.real = a.real - b.real;
        res.image = a.image - b.image;
        return res;
    }


    // @return = a * b + c
    inline _THREAD_CALL_ de::CPf _complex_fma(const de::CPf a, const de::CPf b, const de::CPf c)
    {
        de::CPf res;
        res.real = (a.real * b.real - a.image * b.image) + c.real;
        res.image = (a.real * b.image + a.image * b.real) + c.image;
        return res;
    }


    // @return = a * b - c
    inline _THREAD_CALL_ de::CPf _complex_fms(const de::CPf a, const de::CPf b, const de::CPf c)
    {
        de::CPf res;
        res.real = (a.real * b.real - a.image * b.image) - c.real;
        res.image = (a.real * b.image + a.image * b.real) - c.image;
        return res;
    }
    }
}
}

namespace decx
{
    namespace fft
    {
        typedef void (*__called_ST_func_first) (const float*, double*, const size_t, const uint2);

        typedef void (*__called_ST_func_IFFT_first) (const double*, double*, const size_t, const uint2);

        typedef void (*__called_ST_func) (const double*, double*, const size_t, const size_t, const uint2);
        typedef void (*_called_ST_func) (const double*, double*, const double*, const size_t, const size_t, const uint2);

        typedef void (*__called_ST_IFFT_last) (const double*, float*, const size_t, const size_t, const uint2);
    }
}



#endif