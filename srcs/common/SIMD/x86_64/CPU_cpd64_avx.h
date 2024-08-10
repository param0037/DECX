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


#ifndef CPU_CPD64_AVX_H_
#define CPU_CPD64_AVX_H_

#include "../../basic.h"
#include "../../../modules/core/thread_management/thread_pool.h"
#include "../../Classes/classes_util.h"


#define _expand_CPd64_MM256_(__real, __image) _mm256_setr_pd(__real, __image,       \
                                                             __real, __image)       \



namespace decx
{
    namespace dsp
    {
        namespace CPUK{
        inline _THREAD_CALL_
        __m256d _cp2_mul_cp2_fp64(const __m256d __x, const __m256d __y)
        {
            __m256d rr_ii = _mm256_mul_pd(__x, __y);
            __m256d ri_ir = _mm256_mul_pd(__x, _mm256_permute4x64_pd(__y, 0b10110001));
            // real, real | image, image
            __m256d rr_ii_rep = _mm256_sub_pd(rr_ii, _mm256_permute_pd(rr_ii, 0b0101));
            __m256d ri_ir_rep = _mm256_add_pd(ri_ir, _mm256_permute_pd(ri_ir, 0b0101));

            return _mm256_shuffle_pd(rr_ii_rep, ri_ir_rep, 0);
        }


        inline _THREAD_CALL_
        __m256d _cp2_mul_cp1_fp64(const __m256d __x, const de::CPd __y)
        {
            return _cp2_mul_cp2_fp64(__x, _expand_CPd64_MM256_(__y.real, __y.image));
        }


        inline _THREAD_CALL_
        __m256d _cp2_conjugate_fp64(const __m256d __x)
        {
            return _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(__x),
                _mm256_setr_epi64x(0, 0x8000000000000000, 0, 0x8000000000000000)));
        }


        inline _THREAD_CALL_
        /*
        * @return __x * __y + __z
        */
        __m256d _cp2_fma_cp2_fp64(const __m256d __x, const __m256d __y, const __m256d __z)
        {
            return _mm256_add_pd(_cp2_mul_cp2_fp64(__x, __y), __z);
        }


        inline _THREAD_CALL_
        /*
        * @return __x * __y + __z
        */
        __m256d _cp2_fma_cp1_fp64(const __m256d __x, const de::CPd __y, const __m256d __z)
        {
            return _mm256_add_pd(_cp2_mul_cp1_fp64(__x, __y), __z);
        }


        static _THREAD_CALL_ de::CPd _complex_mul(const de::CPd a, const de::CPd b)
        {
            return {
                fma(a.real, b.real, -1.f * a.image * b.image),
                fma(a.real, b.image, a.image * b.real)
            };
        }


        inline _THREAD_CALL_ de::CPd _complex_add(const de::CPd a, const de::CPd b)
        {
            __m128d res = _mm_add_pd(*((__m128d*)&a), *((__m128d*)&b));
            return *((de::CPd*)&res);
        }


        inline _THREAD_CALL_ de::CPd _complex_sub(const de::CPd a, const de::CPd b)
        {
            __m128d res = _mm_sub_pd(*((__m128d*)&a), *((__m128d*)&b));
            return *((de::CPd*)&res);
        }


        // @return = a * b + c
        inline _THREAD_CALL_ de::CPd _complex_fma(const de::CPd a, const de::CPd b, const de::CPd c)
        {
            return {
                (a.real * b.real - a.image * b.image) + c.real,
                (a.real * b.image + a.image * b.real) + c.image
            };
        }


        // @return = a * b - c
        inline _THREAD_CALL_ de::CPd _complex_fms(const de::CPd a, const de::CPd b, const de::CPd c)
        {
            return {
                (a.real * b.real - a.image * b.image) - c.real,
                (a.real * b.image + a.image * b.real) - c.image
            };
        }
        }
    }
}

#endif
