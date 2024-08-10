/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/



#ifndef _GPU_VECTOR_H_
#define _GPU_VECTOR_H_

#include "../basic.h"
#include "class_utils.h"

namespace de
{
    class _DECX_API_ GPU_Vector
    {
    public:
        GPU_Vector() {}


        virtual size_t Len() const = 0;



        virtual void release() = 0;


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src) = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        ~GPU_Vector() {}
    };


    de::GPU_Vector& CreateGPUVectorRef();


    de::GPU_Vector* CreateGPUVectorPtr();


    de::GPU_Vector& CreateGPUVectorRef(const de::_DATA_TYPES_FLAGS_ _type, const size_t _length);


    de::GPU_Vector* CreateGPUVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, const size_t _length);


    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::Vector& src);


        _DECX_API_ de::DH UnpinMemory(de::Vector& src);
    }
}

#endif