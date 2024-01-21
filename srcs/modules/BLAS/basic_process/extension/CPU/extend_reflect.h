/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _EXTEND_REFLECT_H_
#define _EXTEND_REFLECT_H_


#include "extend_reflect_exec.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/Matrix.h"


namespace decx
{
    namespace bp 
    {
        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (8x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        template <bool _print>
        void _extend_reflect_b32_1D(const float* src, float* dst, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH *handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (4x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        template <bool _print>
        void _extend_reflect_b64_1D(const double* src, double* dst, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (16x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        template <bool _print>
        _DECX_API_ void _extend_reflect_b8_1D(const uint8_t* src, uint8_t* dst, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (8x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        template <bool _print>
        void _extend_reflect_b16_1D(const uint16_t* src, uint16_t* dst, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle);

        
        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _ext : Number of elements to be extented on the four dimensions :
        *       _ext.x -> left; 
                _ext.y -> right; 
                _ext.z -> top; 
                _ext.w -> bottom;
        * @param Wsrc : The Aligned width (8x) of src
        * @param Wdst : The Aligned width (8x) of dst
        * @param _actual_Wsrc : The actual width of src, in element
        */
        template <bool _print>
        void _extend_reflect_b32_2D(const float* src, float* dst, const uint4 _ext, 
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);


        template <bool _print>
        _DECX_API_ void _extend_LR_reflect_b32_2D(const float* src, float* dst, const uint2 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);



        _DECX_API_ void _extend_TB_reflect_b32_2D(float* src, const uint2 _ext, const uint32_t Wsrc, const uint32_t Hsrc);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _ext : Number of elements to be extented on the four dimensions :
        *       _ext.x -> left;
                _ext.y -> right;
                _ext.z -> top;
                _ext.w -> bottom;
        * @param Wsrc : The Aligned width (4x) of src
        * @param Wdst : The Aligned width (4x) of dst
        * @param _actual_Wsrc : The actual width of src, in element
        */
        template <bool _print>
        void _extend_reflect_b64_2D(const double* src, double* dst, const uint4 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _ext : Number of elements to be extented on the four dimensions :
        *       _ext.x -> left;
                _ext.y -> right;
                _ext.z -> top;
                _ext.w -> bottom;
        * @param Wsrc : The Aligned width (16x) of src
        * @param Wdst : The Aligned width (16x) of dst
        * @param _actual_Wsrc : The actual width of src, in element
        */
        template <bool _print>
        _DECX_API_ void _extend_reflect_b8_2D(const uint8_t* src, uint8_t* dst, const uint4 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);


        template <bool _print>
        _DECX_API_ void _extend_LR_reflect_b8_2D(const uint8_t* src, uint8_t* dst, const uint2 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);



        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _ext : Number of elements to be extented on the four dimensions :
        *       _ext.x -> left;
                _ext.y -> right;
                _ext.z -> top;
                _ext.w -> bottom;
        * @param Wsrc : The Aligned width (8x) of src
        * @param Wdst : The Aligned width (8x) of dst
        * @param _actual_Wsrc : The actual width of src, in element
        */
        template <bool _print>
        void _extend_reflect_b16_2D(const uint16_t* src, uint16_t* dst, const uint4 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);


        
        template <bool _print>
        static void _extend1D_reflect(decx::_Vector* src, decx::_Vector* dst, const uint32_t _left, 
            const uint32_t _right, de::DH* handle);



        template <bool _print>
        static void _extend2D_reflect(decx::_Matrix* src, decx::_Matrix* dst, const uint4 _ext, de::DH* handle);


        template <bool _print>
        static void _extend1D_border(decx::_Vector* src, decx::_Vector* dst, const void* _val, const uint32_t _left,
            const uint32_t _right, de::DH* handle);



        template <bool _print>
        static void _extend2D_border(decx::_Matrix* src, decx::_Matrix* dst, const void *_val, const uint4 _ext, de::DH* handle);
    }
}



template <bool _print>
static void decx::bp::_extend1D_reflect(decx::_Vector* src, decx::_Vector* dst, const uint32_t left,
    const uint32_t right, de::DH* handle)
{
    switch (src->_single_element_size)
    {
    case sizeof(double) :
        decx::bp::_extend_reflect_b64_1D<_print>((double*)src->Vec.ptr, (double*)dst->Vec.ptr,
            left, right, src->_length, src->length, handle);
        break;

    case sizeof(float):
        decx::bp::_extend_reflect_b32_1D<_print>((float*)src->Vec.ptr, (float*)dst->Vec.ptr,
            left, right, src->_length, src->length, handle);
        break;

    case sizeof(uint8_t):
        decx::bp::_extend_reflect_b8_1D<_print>((uint8_t*)src->Vec.ptr, (uint8_t*)dst->Vec.ptr,
            left, right, src->_length, src->length, handle);
        break;

    case sizeof(uint16_t) :
        decx::bp::_extend_reflect_b16_1D<_print>((uint16_t*)src->Vec.ptr, (uint16_t*)dst->Vec.ptr,
            left, right, src->_length, src->length, handle);
        break;
    default:
        break;
    }
}



template <bool _print>
static void decx::bp::_extend2D_reflect(decx::_Matrix* src, decx::_Matrix* dst, const uint4 _ext_param, de::DH* handle)
{
    switch (src->get_layout()._single_element_size)
    {
    case sizeof(double) :
        decx::bp::_extend_reflect_b64_2D<_print>((double*)src->Mat.ptr, (double*)dst->Mat.ptr, _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;

        case sizeof(float) :
        decx::bp::_extend_reflect_b32_2D<_print>((float*)src->Mat.ptr, (float*)dst->Mat.ptr, _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;

    case sizeof(uint8_t) :
        decx::bp::_extend_reflect_b8_2D<_print>((uint8_t*)src->Mat.ptr, (uint8_t*)dst->Mat.ptr, _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;

    case sizeof(uint16_t) :
        decx::bp::_extend_reflect_b16_2D<_print>((uint16_t*)src->Mat.ptr, (uint16_t*)dst->Mat.ptr, _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;
    default:
        break;
    }
}




#endif