/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _EXTEND_CONSTANT_H_
#define _EXTEND_CONSTANT_H_



#include "extend_constant_exec.h"
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
        
        void _extend_constant_b32_1D(const float* src, float* dst, const float _val, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (4x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        
        void _extend_constant_b64_1D(const double* src, double* dst, const double _val, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (16x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        
        void _extend_constant_b8_1D(const uint8_t* src, uint8_t* dst, const uint8_t _val, const uint32_t _left, const uint32_t _right,
            const size_t _length_src, const size_t _actual_Lsrc, de::DH* handle);


        /**
        * @param src : Input vector
        * @param dst : Output vector
        * @param _left : Number of elements to be extented on the left
        * @param _right : Number of elements to be extented on the right
        * @param _length_src : The Aligned length (8x) of src
        * @param _actual_Lsrc : The actual length of src, in element
        */
        
        void _extend_constant_b16_1D(const uint16_t* src, uint16_t* dst, const uint16_t _val, const uint32_t _left, const uint32_t _right,
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
        
        void _extend_constant_b32_2D(const float* src, float* dst, const float _val, const uint4 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);


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
        
        void _extend_constant_b64_2D(const double* src, double* dst, const double _val, const uint4 _ext,
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
        
        void _extend_constant_b8_2D(const uint8_t* src, uint8_t* dst, const uint8_t _val, const uint4 _ext,
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
        
        void _extend_constant_b16_2D(const uint16_t* src, uint16_t* dst, const uint16_t _val, const uint4 _ext,
            const uint32_t Wsrc, const uint32_t Wdst, const uint32_t _actual_Wsrc, const uint32_t Hsrc, de::DH* handle);



        
        static void _extend1D_constant(decx::_Vector* src, decx::_Vector* dst, void* _val, const uint32_t _left,
            const uint32_t _right, de::DH* handle);



        
        static void _extend2D_constant(decx::_Matrix* src, decx::_Matrix* dst, void* _val, const uint4 _ext, de::DH* handle);
    }
}




static void decx::bp::_extend1D_constant(decx::_Vector* src, decx::_Vector* dst, void* val, const uint32_t left,
    const uint32_t right, de::DH* handle)
{
    switch (src->_single_element_size)
    {
    case sizeof(double) :
        decx::bp::_extend_constant_b64_1D((double*)src->Vec.ptr, (double*)dst->Vec.ptr, *((double*)val),
            left, right, src->_length, src->length, handle);
        break;

    case sizeof(float):
        decx::bp::_extend_constant_b32_1D((float*)src->Vec.ptr, (float*)dst->Vec.ptr, *((float*)val),
            left, right, src->_length, src->length, handle);
        break;

    case sizeof(uint8_t):
        decx::bp::_extend_constant_b8_1D((uint8_t*)src->Vec.ptr, (uint8_t*)dst->Vec.ptr, *((uint8_t*)val),
            left, right, src->_length, src->length, handle);
        break;

    case sizeof(uint16_t) :
        decx::bp::_extend_constant_b16_1D((uint16_t*)src->Vec.ptr, (uint16_t*)dst->Vec.ptr, *((uint16_t*)val),
            left, right, src->_length, src->length, handle);
        break;
    default:
        break;
    }
}





static void decx::bp::_extend2D_constant(decx::_Matrix* src, decx::_Matrix* dst, void* val, const uint4 _ext_param, de::DH* handle)
{
    switch (src->get_layout()._single_element_size)
    {
    case sizeof(double) :
        decx::bp::_extend_constant_b64_2D((double*)src->Mat.ptr, (double*)dst->Mat.ptr, *((double*)val), _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;

        case sizeof(float) :
        decx::bp::_extend_constant_b32_2D((float*)src->Mat.ptr, (float*)dst->Mat.ptr, *((float*)val), _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;

    case sizeof(uint8_t) :
        decx::bp::_extend_constant_b8_2D((uint8_t*)src->Mat.ptr, (uint8_t*)dst->Mat.ptr, *((uint8_t*)val), _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;

    case sizeof(uint16_t) :
        decx::bp::_extend_constant_b16_2D((uint16_t*)src->Mat.ptr, (uint16_t*)dst->Mat.ptr, *((uint16_t*)val), _ext_param, src->Pitch(), dst->Pitch(),
            src->Width(), src->Height(), handle);
        break;
    default:
        break;
    }
}



#endif