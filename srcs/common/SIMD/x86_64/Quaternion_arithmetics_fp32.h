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

#ifndef _QUATERNION_ARITHMETICS_FP32_H_
#define _QUATERNION_ARITHMETICS_FP32_H_


#include "../../decx_utils_macros.h"
#include "../../vector_defines.h"
#include "../intrinsics_ops.h"


namespace decx
{
namespace utils {
namespace simd {
    
    __m128 quaternion_mul_fp32(__m128 q1, __m128 q2)
    {
        __m128 res = _mm_mul_ps(q2, _mm_permute_ps(q1, 0b00));

        __m128 tmp = _mm_xor_ps(_mm_permute_ps(q2, 0b10110001),
                _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0, 0x80000000, 0)));
        res = _mm_fmadd_ps(tmp, _mm_permute_ps(q1, 0b01010101), res);

        tmp = _mm_xor_ps(_mm_permute_ps(q2, 0b01001110),
                _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0, 0, 0x80000000)));
        res = _mm_fmadd_ps(tmp, _mm_permute_ps(q1, 0b10101010), res);

        tmp = _mm_xor_ps(_mm_permute_ps(q2, 0b00011011),
                _mm_castsi128_ps(_mm_setr_epi32(0x80000000, 0x80000000, 0, 0)));
        res = _mm_fmadd_ps(tmp, _mm_permute_ps(q1, 0b11111111), res);

        return res;
    }


    __m128 quaternion_conj_fp32(__m128 q)
    {
        return _mm_xor_ps(q, _mm_castsi128_ps(_mm_setr_epi32(0, 0x80000000, 0x80000000, 0x80000000)));
    }


    float quaternion_mag_fp32(__m128 q)
    {
        return decx::utils::simd::_mm128_h_sum(_mm_mul_ps(q, q));
    }

//     __m128 quaternion_rcp_fp32(__m128 q)
//     {
//         __m128 q_sq = _mm_mul_ps(q, q);
//         __m128 hadd = _mm_hadd_ps(q_sq, q_sq);
//         __m128 mag_v4 = _mm_add_ps(hadd, _mm_permute_ps(hadd, 0b10110010));
//         // Need sqrt
//     }
}
}
}

#endif
