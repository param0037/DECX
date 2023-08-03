/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DECX_UTILS_FUNCTIONS_H_
#define _DECX_UTILS_FUNCTIONS_H_

#include "decx_utils_macros.h"
#include "../vector_defines.h"
#ifdef _DECX_CPU_PARTS_
#include "intrinsics_ops.h"
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
        static int _GetHighest_abd(size_t __x) noexcept;

        /*
        * @brief : The '_abd' suffix means that the return value WILL
        * plus one when the input value reaches the critical points (e.g.
        * 2, 4, 8, 16, 32...)
        */
        static int _GetHighest(size_t __x) noexcept;


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



//#ifdef _DECX_CUDA_PARTS_
//        template <typename _Ty>
//        __device__
//        _Ty cu_ceil(_Ty __deno, _Ty __numer) {
//            return (__deno / __numer) + (int)((bool)(__deno % __numer));
//        }
//#endif


        constexpr inline static int Iabs(int n) noexcept {
            return (n ^ (n >> 31)) - (n >> 31);
        }

        
        /**
         * NOrmally, src_len should not larger than dst_len
        */
        template <size_t dst_len>
        static void decx_strcpy(char (&dst)[dst_len], const char* src);

        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param __x : Offset along height
        * @param __y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _In_Type, typename _Out_Type>
        static _Out_Type* ptr_shift_xy(_In_Type* _initial_ptr, const size_t __x, const size_t __y, const size_t _pitch);

        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param offset_xy : ~.x : Offset along height; ~.y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _In_Type, typename _Out_Type>
        static _Out_Type* ptr_shift_xy(_In_Type* _initial_ptr, const uint2 offset_xy, const size_t _pitch);


        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param __x : Offset along height
        * @param __y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _Ty>
        static void ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint __x, const uint __y, const size_t _pitch);

        /*
        * @param _initial_ptr : The pointer where start to offset
        * @param offset_xy : ~.x : Offset along height; ~.y : Offset along width
        * @param _pitch : The width of the virtal square
        */
        template <typename _Ty>
        static void ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint2 offset_xy, const size_t _pitch);
    }
}




static int decx::utils::_GetHighest_abd(size_t __x) noexcept
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



static int _GetHighest(size_t __x) noexcept
{
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}



static int decx::utils::_GetHighest(size_t __x) noexcept
{
    int res = 0;
    while (__x) {
        ++res;
        __x >>= 1;
    }
    return res;
}




template <size_t dst_len>
static void decx::utils::decx_strcpy(char (&dst)[dst_len], const char* src)
{
#ifdef Windows
    strcpy_s<dst_len>(dst, src);
#endif

#ifdef Linux
    memcpy(dst, src, strlen(src) * sizeof(char));
#endif
}


template <typename _In_Type, typename _Out_Type> static FORCEINLINE
_Out_Type* decx::utils::ptr_shift_xy(_In_Type* _initial_ptr, const size_t __x, const size_t __y, const size_t _pitch)
{
    return reinterpret_cast<_Out_Type*>(_initial_ptr) + (size_t)__x * _pitch + (size_t)__y;
}



template <typename _In_Type, typename _Out_Type> static FORCEINLINE
_Out_Type* decx::utils::ptr_shift_xy(_In_Type* _initial_ptr, const uint2 offset_xy, const size_t _pitch)
{
    return reinterpret_cast<_Out_Type*>(_initial_ptr) + (size_t)offset_xy.x * _pitch + (size_t)offset_xy.y;
}



template <typename _Ty> static FORCEINLINE
void decx::utils::ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint __x, const uint __y, const size_t _pitch)
{
    *_initial_ptr += (size_t)__x * _pitch + (size_t)__y;
}



template <typename _Ty> static FORCEINLINE
void decx::utils::ptr_shift_xy_inplace(_Ty** _initial_ptr, const uint2 offset_xy, const size_t _pitch)
{
    *_initial_ptr += (size_t)offset_xy.x * _pitch + (size_t)offset_xy.y;
}




#define DECX_PTR_SHF_XY decx::utils::ptr_shift_xy
#define DECX_PTR_SHF_XY_INP decx::utils::ptr_shift_xy_inplace

#define DECX_PTR_SHF_XY_SAME_TYPE(_src_ptr, _offset_x, _offset_y, _pitch) \
    (_src_ptr) + (_offset_x) * (_pitch) + (_offset_y)


#endif