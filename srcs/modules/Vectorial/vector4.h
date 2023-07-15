/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _VECTOR4_H_
#define _VECTOR4_H_


#include "../core/basic.h"
#include "../core/utils/intrinsics_ops.h"
#include "../core/thread_management/thread_pool.h"


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

namespace decx {
#ifdef _DECX_CPU_PARTS_
    union __align__(16) _Vector4f {
        float _vec_sp[4];
        __m128 _vec;
    };

    union __align__(16) _Vector8f {
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
#endif

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
    __m128 tmp;
    tmp = _mm_mul_ps(__x._vec, __x._vec);
    float length = decx::utils::simd::_mm128_h_sum(tmp);
    decx::_Vector4f res;
    res._vec = _mm_div_ps(__x._vec, _mm_set1_ps(sqrt(length)));
    return res;
}


_THREAD_GENERAL_ static inline float 
decx::vec::vector4_dot(const decx::_Vector4f __x, const decx::_Vector4f __y)
{
    __m128 tmp = _mm_mul_ps(__x._vec, __y._vec);
    return decx::utils::simd::_mm128_h_sum(tmp);
}



_THREAD_GENERAL_ static void 
decx::mat::_mat4x4_transpose_fp32(decx::_Mat4x4f* __x)
{
    __m128 _Tmp3, _Tmp2, _Tmp1, _Tmp0;
    _Tmp0 = _mm_shuffle_ps(__x->_row[0], __x->_row[1], 0x44);
    _Tmp2 = _mm_shuffle_ps(__x->_row[0], __x->_row[1], 0xEE);
    _Tmp1 = _mm_shuffle_ps(__x->_row[2], __x->_row[3], 0x44);
    _Tmp3 = _mm_shuffle_ps(__x->_row[2], __x->_row[3], 0xEE);
        
    __x->_row[0] = _mm_shuffle_ps(_Tmp0, _Tmp1, 0x88);
    __x->_row[1] = _mm_shuffle_ps(_Tmp0, _Tmp1, 0xDD);
    __x->_row[2] = _mm_shuffle_ps(_Tmp2, _Tmp3, 0x88);
    __x->_row[3] = _mm_shuffle_ps(_Tmp2, _Tmp3, 0xDD);
}


_THREAD_GENERAL_ static inline 
decx::_Vector4f vector4_cross3(const decx::_Vector4f __x, const decx::_Vector4f __y)
{
    decx::_Vector4f ans;
    ans._vec = _mm_mul_ps(_mm_permute_ps(__x._vec, 0b11001001), _mm_permute_ps(__y._vec, 0b11010010));
    __m128 tmp = _mm_mul_ps(_mm_permute_ps(__y._vec, 0b11001001), _mm_permute_ps(__x._vec, 0b11010010));
    ans._vec = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(_mm_sub_ps(ans._vec, tmp)), 
        _mm_setr_epi32(0xffffffff, 0xffffffff, 0xffffffff, 0x0)));
    return ans;
}


#endif

#endif