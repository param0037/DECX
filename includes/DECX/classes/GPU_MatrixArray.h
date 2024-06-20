/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _GPU_MATRIXARRAY_H_
#define _GPU_MATRIXARRAY_H_

#include "../basic.h"
#include "MatrixArray.h"



namespace de
{
    class _DECX_API_ GPU_MatrixArray
    {
    public:
        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint MatrixNumber() const = 0;


        virtual de::GPU_MatrixArray& SoftCopy(de::GPU_MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };


    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef();


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr();


    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint32_t _array_length);


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint32_t _array_length);


    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::MatrixArray& src);


        _DECX_API_ de::DH UnpinMemory(de::MatrixArray& src);
    }
}


#endif