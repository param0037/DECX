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


#ifndef _VECTOR4_H_
#define _VECTOR4_H_


#include "../../../common/basic.h"
#include "../../../common/SIMD/intrinsics_ops.h"
#ifdef _DECX_CPU_PARTS_
#include "../../core/thread_management/thread_pool.h"
#endif


// external

namespace de
{
    struct __align__(16) Vector3f {
        float x, y, z;
    };

    struct __align__(8) Vector2f {
        float x, y;
    };

    struct __align__(16) Vector4f {
        float x, y, z, w;
    };
}


// internal

namespace decx 
{
#ifdef _DECX_CPU_PARTS_
#if defined(__x86_64__) || defined(__i386__)
    union __align__(16) _Vector4f {
        float _vec_sp[4];
        __m128 _vec;
    };

    union __align__(32) _Vector8f {
        float _vec_sp[8];
        __m256 _vec;
    };

    union __align__(8) _Vector2f {
        float _vec_sp[2];
    };


    union __align__(64) _Mat4x4f {
        __m128 _row[4];
        __m128 _col[4];
    };

#elif defined(__aarch64__) || defined(__arm__)

    union __align__(16) _Vector4f {
        float _vec_sp[4];
        float32x4_t _vec;
    };

    union __align__(16) _Vector8f {
        float _vec_sp[8];
        float32x4x2_t _vec;
    };

    union __align__(8) _Vector2f {
        float32x2_t _vec;
        float _vec_sp[2];
    };


    union __align__(64) _Mat4x4f {
        float32x4_t _row[4];
        float32x4_t _col[4];
    };
#endif  // #if defined(__x86_x64__) || defined(__i386__)

    namespace vec {
        /**
        * Normalize the vector, that is, turn it into a dirctional vector
        */
        _THREAD_GENERAL_ static inline decx::_Vector4f vector4_nomalize(const decx::_Vector4f __x);

        /**
        * Apply __x DOT __y
        */
        _THREAD_GENERAL_ static inline float vector4_dot(const decx::_Vector4f __x, const decx::_Vector4f __y);

        /**
        * Treat decx::_vector4 as a three dimensional vector and res = __x X __y
        */
        _THREAD_GENERAL_ static inline decx::_Vector4f vector4_cross3(const decx::_Vector4f __x, const decx::_Vector4f __y);
    }

    namespace mat {
        _THREAD_GENERAL_ static void _mat4x4_transpose_fp32(decx::_Mat4x4f* __x);
    }
#endif      // #ifdef _DECX_CPU_PARTS_

#ifdef _DECX_CUDA_PARTS_
    union __align__(16) _Vector4f {
        float4 _vec;
    };


    union __align__(8) _Vector2f {
        float2 _vec;
    };
#endif
}


#ifdef _DECX_CPU_PARTS_

_THREAD_GENERAL_ static inline decx::_Vector4f 
decx::vec::vector4_nomalize(const decx::_Vector4f __x)
{
#if defined(__x86_64__) || defined(__i386__)
    __m128 tmp;
    tmp = _mm_mul_ps(__x._vec, __x._vec);
    float length = decx::utils::simd::_mm128_h_sum(tmp);
    decx::_Vector4f res;
    res._vec = _mm_div_ps(__x._vec, _mm_set1_ps(sqrt(length)));
    return res;
#elif defined(__aarch64__) || defined(__arm__)

#endif
}


_THREAD_GENERAL_ static inline float 
decx::vec::vector4_dot(const decx::_Vector4f __x, const decx::_Vector4f __y)
{
#if defined(__x86_64__) || defined(__i386__)
    __m128 tmp = _mm_mul_ps(__x._vec, __y._vec);
    return decx::utils::simd::_mm128_h_sum(tmp);
#elif defined(__aarch64__) || defined(__arm__)

#endif
}



_THREAD_GENERAL_ static void 
decx::mat::_mat4x4_transpose_fp32(decx::_Mat4x4f* __x)
{
#if defined(__x86_64__) || defined(__i386__)
    __m128 _Tmp3, _Tmp2, _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(__x->_row[0], __x->_row[1], 0x44);
    _Tmp2 = _mm_shuffle_ps(__x->_row[0], __x->_row[1], 0xEE);
    _Tmp1 = _mm_shuffle_ps(__x->_row[2], __x->_row[3], 0x44);
    _Tmp3 = _mm_shuffle_ps(__x->_row[2], __x->_row[3], 0xEE);
        
    __x->_row[0] = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
    __x->_row[1] = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
    __x->_row[2] = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88);
    __x->_row[3] = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD);
#elif defined(__aarch64__) || defined(__arm__)

#endif
}


_THREAD_GENERAL_ static inline 
decx::_Vector4f vector4_cross3(const decx::_Vector4f __x, const decx::_Vector4f __y)
{
#if defined(__x86_64__) || defined(__i386__)
    decx::_Vector4f ans;
    ans._vec = _mm_mul_ps(_mm_permute_ps(__x._vec, 0b11001001), _mm_permute_ps(__y._vec, 0b11010010));
    __m128 tmp = _mm_mul_ps(_mm_permute_ps(__y._vec, 0b11001001), _mm_permute_ps(__x._vec, 0b11010010));
    ans._vec = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(_mm_sub_ps(ans._vec, tmp)), 
        _mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0x0)));
    return ans;
#elif defined(__aarch64__) || defined(__arm__)

#endif
}


#endif      // #ifdef _DECX_CPU_PARTS_

#endif