/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _VECTOR_H_
#define _VECTOR_H_

#include "../basic.h"
#include "type_info.h"

namespace de
{
    class _DECX_API_ Vector
    {
    public:
        Vector() {}


        virtual size_t Len() const = 0;


        virtual float* ptr_fp32(size_t index) = 0;
        virtual int* ptr_int32(size_t index) = 0;
        virtual uint64_t* ptr_uint64(size_t index) = 0;
        virtual double* ptr_fp64(size_t index) = 0;
        virtual de::Half* ptr_fp16(size_t index) = 0;
        virtual de::CPf* ptr_cpl32(size_t index) = 0;
        virtual uint8_t* ptr_uint8(size_t index) = 0;
        virtual de::Vector4f* ptr_vec4f(size_t index) = 0;

        virtual void release() = 0;


        virtual de::Vector& SoftCopy(de::Vector& src) = 0;

        virtual int Type() const = 0;


        ~Vector() {}
    };


    _DECX_API_ de::Vector& CreateVectorRef();


    _DECX_API_ de::Vector* CreateVectorPtr();


    _DECX_API_ de::Vector& CreateVectorRef(const de::_DATA_TYPES_FLAGS_ _type, size_t len);


    _DECX_API_ de::Vector* CreateVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, size_t len);
}




#endif