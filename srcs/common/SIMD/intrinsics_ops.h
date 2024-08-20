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


#ifndef _INTRINSICS_OPS_H_
#define _INTRINSICS_OPS_H_


#include "../decx_utils_macros.h"
#include "../vector_defines.h"

#if defined(__x86_64__) || defined(__i386__)
#include "x86_64/simd_fast_math_avx2.h"
#include "x86_64/simd_fast_math_sse.h"
#endif


namespace decx
{
namespace utils
{
#ifdef _DECX_CPU_PARTS_
namespace simd 
{
    typedef union __align__(8) xmm64_reg {
#if defined(__x86_64__) || defined(__i386__)
        __m64       _m64;
#endif
#if defined(__aarch64__) || defined(__arm__)
        float32x2_t _vf;
        int32x2_t   _vi;
        int8x8_t    _vuc;
#endif
        double      _fp64;
        uint64_t    _ull;
        float       _arrf[2];
    }_mmv64;


    typedef union __align__(16) xmm128_reg {
#if defined(__x86_64__) || defined(__i386__)
        __m128      _vf;
        __m128d     _vd;
        __m128i     _vi;

        inline void zeros() {
            this->_vi = _mm_xor_si128(this->_vi, this->_vi);
        }
#endif
#if defined(__aarch64__) || defined(__arm__)
        float32x4_t _vf;
        int32x4_t   _vi;
        uint32x4_t  _vui;
        float64x2_t _vd;
        int8x16_t   _vuc;
        int16x8_t   _vs;
        uint8x16_t  _vus;

        inline void zeros(){
            this->_vui = veorq_u32(this->_vui, this->_vui);
        }
#endif
        float       _arrf[4];
        int32_t     _arri[4];
        int16_t     _arrs[8];
        double      _arrd[2];
        uint64_t    _arrull[2];
    }_mmv128;


    typedef union __align__(32) xmm256_reg 
    {
        float       _arrf[8];
        double      _arrd[4];
        uint64_t    _arrull[4];

        decx::utils::simd::xmm128_reg _vmm128[2];

#if defined(__x86_64__) || defined(__i386__)
        __m256      _vf;
        __m256d     _vd;
        __m256i     _vi;

        __m128 _vf2[2];
        __m128d _vd2[2];


        inline float read_fp32_0() {
            return _mm256_cvtss_f32(this->_vf);
        }

        inline double read_fp64_0() {
            return _mm256_cvtsd_f64(this->_vd);
        }

        /*inline float read_fp32_123(const int idx) {
            return _mm_extract_ps(_mm256_castps256_ps128(this->_vf), idx);
        }*/


        inline void zeros(){
            this->_vi = _mm256_xor_si256(this->_vi, this->_vi);
        }
#endif
        

#if defined(__aarch64__) || defined(__arm__)
        float32x4x2_t   _vf;
        int32x4x2_t     _vi;
        float64x2x2_t   _vd;
        uint8x16x2_t    _vuc;
        uint32x4x2_t    _vui;
        uint16x8x2_t    _vus;

        
        inline void zeros(){
            this->_vui.val[0] = veorq_u32(this->_vui.val[0], this->_vui.val[0]);
            this->_vui.val[1] = veorq_u32(this->_vui.val[1], this->_vui.val[1]);
        }
#endif
    }_mmv256;

#if defined(__x86_64__) || defined(__i386__)
    static float _mm128_h_sum(__m128 v);


    static float _mm256_h_sum(__m256 v);

    static float _mm256_h_max(__m256 v);

    static float _mm256_h_min(__m256 v);


    static double _mm256d_h_sum(__m256d v);
    static int64_t _mm256i_h_sum_epi64(__m256i v);

    static double _mm256d_h_max(__m256d v);

    static double _mm256d_h_min(__m256d v);

    static uint8_t _mm128_h_max_u8(__m128i v);

    static uint8_t _mm128_h_min_u8(__m128i v);


    /**
    * The data move from higher address to lower address for 1 element
    * @param __proc : the pointer of the value to be processed
    */
    inline __m256 _mm256_shift1_H2L(__m256 __proc);


    /**
    * The data move from higher address to lower address for 2 elements
    * @param __proc : the pointer of the value to be processed
    */
    inline __m256 _mm256_shift2_H2L(__m256 __proc);


    /**
    * The data move from higher address to lower address for 1 element
    * @param __proc : the pointer of the value to be processed
    */
    inline __m256 _mm256_shift1_L2H(__m256 __proc);


    /**
    * The data move from higher address to lower address for 2 elements
    * @param __proc : the pointer of the value to be processed
    */
    inline __m256 _mm256_shift2_L2H(__m256 __proc);
#endif

#if defined(__aarch64__) || defined(__arm__)
    inline float32x4_t vswap_middle_f32(float32x4_t __src)
    {
        return vzip1q_f32(__src, vextq_f32(__src, __src, 2));
    }

    
    inline decx::utils::simd::xmm128_reg& vdupq_n_zeros(decx::utils::simd::xmm128_reg& reg)
    {
        reg._vui = veorq_u32(reg._vui, reg._vui);
        return reg;
    }

#endif  // #if defined(__x86_x64__) || defined(__i386__)

    
}
#endif  // #ifdef _DECX_CPU_PARTS_
}
}


#ifdef _DECX_CPU_PARTS_
#if defined(__x86_64__) || defined(__i386__)
static float decx::utils::simd::_mm128_h_sum(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_extract_ps(sums, 0);
}


static double decx::utils::simd::_mm256d_h_sum(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    int64_t res = _mm_extract_epi64(_mm_castpd_si128(_mm_add_sd(vlow, high64)), 0);  // reduce to scalar
    return *((double*)&res);
}


static int64_t decx::utils::simd::_mm256i_h_sum_epi64(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extractf128_si256(v, 1); // high 128
    vlow = _mm_add_epi64(vlow, vhigh);     // reduce down to 128

    __m128i high64 = _mm_unpackhi_epi64(vlow, vlow);
    return _mm_extract_epi64(_mm_add_epi64(vlow, high64), 0);
}



static double decx::utils::simd::_mm256d_h_max(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_max_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    int64_t res = _mm_extract_epi64(_mm_castpd_si128(_mm_max_sd(vlow, high64)), 0);  // reduce to scalar
    return *((double*)&res);
}


static double decx::utils::simd::_mm256d_h_min(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_min_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    int64_t res = _mm_extract_epi64(_mm_castpd_si128(_mm_min_sd(vlow, high64)), 0);  // reduce to scalar
    return *((double*)&res);
}


static uint8_t decx::utils::simd::_mm128_h_max_u8(__m128i v) {
    __m128i _halv_2 = _mm_max_epu8(v, _mm_shuffle_epi32(v, 0b01001110));
    __m128i _halv_4 = _mm_max_epu8(_halv_2, _mm_shuffle_epi32(_halv_2, 0b10110001));
    __m128i _halv_8 = _mm_max_epu8(_halv_4, _mm_shuffle_epi8(_halv_4, _mm_set1_epi32(0x01000302)));
    __m128i _halv_16 = _mm_max_epu8(_halv_8, _mm_shuffle_epi8(_halv_8, _mm_set1_epi16(0x0001)));

    return _mm_extract_epi8(_halv_16, 0);
}



static uint8_t decx::utils::simd::_mm128_h_min_u8(__m128i v) {
    __m128i _halv_2 = _mm_min_epu8(v, _mm_shuffle_epi32(v, 0b01001110));
    __m128i _halv_4 = _mm_min_epu8(_halv_2, _mm_shuffle_epi32(_halv_2, 0b10110001));
    __m128i _halv_8 = _mm_min_epu8(_halv_4, _mm_shuffle_epi8(_halv_4, _mm_set1_epi32(0x01000302)));
    __m128i _halv_16 = _mm_min_epu8(_halv_8, _mm_shuffle_epi8(_halv_8, _mm_set1_epi16(0x0001)));

    return _mm_extract_epi8(_halv_16, 0);
}



static float decx::utils::simd::_mm256_h_sum(__m256 x) {
    __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    __m128 loQuad = _mm256_castps256_ps128(x);
    __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    hiQuad = _mm_shuffle_ps(sumQuad, sumQuad, _MM_SHUFFLE(2, 3, 0, 1));
    sumQuad = _mm_add_ps(sumQuad, hiQuad);
    loQuad = _mm_movehl_ps(hiQuad, sumQuad);
    sumQuad = _mm_add_ss(sumQuad, loQuad);
    return _mm_extract_ps(sumQuad, 0);
}



static float decx::utils::simd::_mm256_h_max(__m256 x) {
    __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    __m128 loQuad = _mm256_castps256_ps128(x);
    __m128 sumQuad = _mm_max_ps(loQuad, hiQuad);
    hiQuad = _mm_shuffle_ps(sumQuad, sumQuad, _MM_SHUFFLE(2, 3, 0, 1));
    sumQuad = _mm_max_ps(sumQuad, hiQuad);
    loQuad = _mm_movehl_ps(hiQuad, sumQuad);
    sumQuad = _mm_max_ss(sumQuad, loQuad);
    return _mm_extract_ps(sumQuad, 0);
}



static float decx::utils::simd::_mm256_h_min(__m256 x) {
    __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    __m128 loQuad = _mm256_castps256_ps128(x);
    __m128 sumQuad = _mm_min_ps(loQuad, hiQuad);
    hiQuad = _mm_shuffle_ps(sumQuad, sumQuad, _MM_SHUFFLE(2, 3, 0, 1));
    sumQuad = _mm_min_ps(sumQuad, hiQuad);
    loQuad = _mm_movehl_ps(hiQuad, sumQuad);
    sumQuad = _mm_min_ss(sumQuad, loQuad);
    return _mm_extract_ps(sumQuad, 0);
}



inline __m256 decx::utils::simd::_mm256_shift1_H2L(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(0, 3, 2, 1));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 81);
    return _mm256_blend_ps(tmp0, tmp1, 0x88);
}


inline __m256 decx::utils::simd::_mm256_shift2_H2L(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 81);
    return _mm256_blend_ps(tmp0, tmp1, 0xcc);
}


inline __m256 decx::utils::simd::_mm256_shift1_L2H(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(2, 1, 0, 3));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 41);
    return _mm256_blend_ps(tmp0, tmp1, 0x11);
}


inline __m256 decx::utils::simd::_mm256_shift2_L2H(__m256 __proc)
{
    __m256 tmp0 = _mm256_permute_ps(__proc, _MM_SHUFFLE(1, 0, 3, 2));
    __m256 tmp1 = _mm256_permute2f128_ps(tmp0, tmp0, 41);
    return _mm256_blend_ps(tmp0, tmp1, 0x33);
}



// -------------------------------------- _mm256d ---------------------------------------


namespace decx
{
    namespace utils
    {
#ifdef _DECX_CPU_PARTS_
        namespace simd {
            inline __m256d _mm256d_shift1_H2L(__m256d _proc);
        }
#endif
    }
}


inline 
__m256d decx::utils::simd::_mm256d_shift1_H2L(__m256d _proc) {
    __m256d tmp = _mm256_permute_pd(_proc, 0b0101);
    _proc = _mm256_permute2f128_pd(tmp, tmp, 1);
    return _mm256_blend_pd(tmp, _proc, 0b1010);
}

#elif defined(__aarch64__) || defined(__arm__)

#endif      // #if defined(__x86_64__) || defined(__i386__)

#endif      // #ifdef _DECX_CPU_PARTS_

#endif
