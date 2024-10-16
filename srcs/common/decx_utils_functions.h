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


#ifndef _DECX_UTILS_FUNCTIONS_H_
#define _DECX_UTILS_FUNCTIONS_H_

#include "decx_utils_macros.h"
#include "vector_defines.h"
#include "string.h"
#ifdef _DECX_CPU_PARTS_
#include "SIMD/intrinsics_ops.h"
#endif


namespace decx
{
    namespace utils
    {
        /*
        * @brief : The '_abd' suffix means that the return value WILL NOT
        * plus one when the input value reaches the critical points (e.g.
        * 2, 4, 8, 16, 32...)
        */
        static int _GetHighest_abd(uint64_t __x) noexcept;

        /*
        * @brief : The '_abd' suffix means that the return value WILL
        * plus one when the input value reaches the critical points (e.g.
        * 2, 4, 8, 16, 32...)
        */
        static int _GetHighest(uint64_t __x) noexcept;


        /*
        * @return return __x < _boundary ? _boundary : __x;
        */
        template <typename _Ty>
        constexpr static _Ty clamp_min(_Ty __x, _Ty _bpunary) noexcept;
        

        /*
        * @return return __x > _boundary ? _boundary : __x;
        */
        template <typename _Ty>
        constexpr static _Ty clamp_max(_Ty __x, _Ty _bpunary) noexcept;


        /*
        * @return return (__deno % __numer) != 0 ? __deno / __numer + 1 : __deno / __numer;
        */
        template <typename _Ty>
#ifdef _DECX_CUDA_PARTS_
__host__ __device__
#endif
        constexpr
        inline static _Ty ceil(_Ty __deno, _Ty __numer) noexcept;


template <typename _Ty>
#ifdef _DECX_CUDA_PARTS_
__host__ __device__
#endif
        /**
        * @brief Only valid for positive int32_t, uint32_t, positive int64_t, and uint64_t;
        */
        constexpr
        inline static _Ty fast_uint_ceil2(_Ty __src) noexcept;


        template <typename _Ty>
#ifdef _DECX_CUDA_PARTS_
        __host__ __device__
#endif
        constexpr inline static _Ty align(_Ty __x, const uint32_t _alignment);
        


        constexpr inline static int Iabs(int n) noexcept {
            return (n ^ (n >> 31)) - (n >> 31);
        }

        
        /**
         * NOrmally, src_len should not larger than dst_len
        */
        template <uint64_t dst_len>
        static void decx_strcpy(char (&dst)[dst_len], const char* src);

        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param __x : Offset along height
        * @param __y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _In_Type, typename _Out_Type>
        static _Out_Type* ptr_shift_xy(_In_Type* _initial_ptr, const uint64_t __x, const uint64_t __y, const uint64_t _pitch);

        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param offset_xy : ~.x : Offset along height; ~.y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _In_Type, typename _Out_Type>
        static _Out_Type* ptr_shift_xy(_In_Type* _initial_ptr, const uint2 offset_xy, const uint64_t _pitch);


        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param __x : Offset along height
        * @param __y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _Ty>
        static void ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint __x, const uint __y, const uint64_t _pitch);

        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param offset_xy : ~.x : Offset along height; ~.y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _Ty>
        static void ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint2 offset_xy, const uint64_t _pitch);
    }
}




static int decx::utils::_GetHighest_abd(uint64_t __x) noexcept
{
    --__x;
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}


/*
* @return return __x < _boundary ? _boundary : __x;
*/
template <typename _Ty>
constexpr static _Ty decx::utils::clamp_min(_Ty __x, _Ty _boundary) noexcept{
    return __x < _boundary ? _boundary : __x;
}

/*
* @return return __x > _boundary ? _boundary : __x;
*/
template <typename _Ty>
constexpr static _Ty decx::utils::clamp_max(_Ty __x, _Ty _boundary) noexcept {
    return __x > _boundary ? _boundary : __x;
}


template <typename _Ty>
#ifdef _DECX_CUDA_PARTS_
__host__ __device__
#endif
constexpr
inline static _Ty decx::utils::ceil(_Ty __deno, _Ty __numer) noexcept
{
    return (__deno / __numer) + (_Ty)((bool)(__deno % __numer));
}


template <typename _Ty>
#ifdef _DECX_CUDA_PARTS_
__host__ __device__
#endif
constexpr
inline static _Ty decx::utils::fast_uint_ceil2(_Ty __src) noexcept
{
    return ((__src >> 1) + (__src & 1));
}


template <typename _Ty>
#ifdef _DECX_CUDA_PARTS_
__host__ __device__
#endif
constexpr static inline _Ty
decx::utils::align(_Ty __x, const uint32_t _alignment)
{
    return decx::utils::ceil<_Ty>(__x, _alignment) * _alignment;
}


static int _GetHighest(uint64_t __x) noexcept
{
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}



static int decx::utils::_GetHighest(uint64_t __x) noexcept
{
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}




template <uint64_t dst_len>
static void decx::utils::decx_strcpy(char (&dst)[dst_len], const char* src)
{
#ifdef _MSC_VER
    strcpy_s<dst_len>(dst, src);
#endif

#ifdef __GNUC__
    strcpy(dst, src);
    //memcpy(dst, src, dst_len * sizeof(char));
#endif
}


template <typename _In_Type, typename _Out_Type> static FORCEINLINE
_Out_Type* decx::utils::ptr_shift_xy(_In_Type* _initial_ptr, const uint64_t __y, const uint64_t __x, const uint64_t _pitch)
{
    return reinterpret_cast<_Out_Type*>(_initial_ptr) + (uint64_t)__y * _pitch + (uint64_t)__x;
}



template <typename _In_Type, typename _Out_Type> static FORCEINLINE
_Out_Type* decx::utils::ptr_shift_xy(_In_Type* _initial_ptr, const uint2 offset_xy, const uint64_t _pitch)
{
    return reinterpret_cast<_Out_Type*>(_initial_ptr) + (uint64_t)offset_xy.x * _pitch + (uint64_t)offset_xy.y;
}



template <typename _Ty> static FORCEINLINE
void decx::utils::ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint __x, const uint __y, const uint64_t _pitch)
{
    *_initial_ptr += (uint64_t)__x * _pitch + (uint64_t)__y;
}



template <typename _Ty> static FORCEINLINE
void decx::utils::ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint2 offset_xy, const uint64_t _pitch)
{
    *_initial_ptr += (uint64_t)offset_xy.x * _pitch + (uint64_t)offset_xy.y;
}




#define DECX_PTR_SHF_XY decx::utils::ptr_shift_xy
#define DECX_PTR_SHF_XY_INP decx::utils::ptr_shift_xy_inplace

#define DECX_PTR_SHF_XY_SAME_TYPE(_src_ptr, _offset_y, _offset_x, _pitch) \
    (_src_ptr) + (_offset_y) * (_pitch) + (_offset_x)


#endif
