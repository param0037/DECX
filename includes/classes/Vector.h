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


        virtual size_t Len() = 0;


        virtual float* ptr_fp32(size_t index) = 0;
        virtual int* ptr_int32(size_t index) = 0;
        virtual double* ptr_fp64(size_t index) = 0;
        virtual de::Half* ptr_fp16(size_t index) = 0;
        virtual de::CPf* ptr_cpl32(size_t index) = 0;
        virtual uint8_t* ptr_uint8(size_t index) = 0;


        virtual void release() = 0;


        virtual de::Vector& operator=(de::Vector& src) = 0;


        virtual ~Vector() {}
    };


    _DECX_API_ de::Vector& CreateVectorRef();


    _DECX_API_ de::Vector* CreateVectorPtr();


    _DECX_API_ de::Vector& CreateVectorRef(const int _type, size_t len, const int flag);


    _DECX_API_ de::Vector* CreateVectorPtr(const int _type, size_t len, const int flag);
}

#endif