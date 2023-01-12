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

namespace de
{
    class _DECX_API_ GPU_Vector
    {
    public:
        GPU_Vector() {}


        virtual size_t Len() = 0;


        virtual void load_from_host(de::Vector& src) = 0;


        virtual void load_to_host(de::Vector& dst) = 0;


        virtual void release() = 0;


        virtual de::GPU_Vector& operator=(de::GPU_Vector& src) = 0;


        virtual int Type() = 0;


        ~GPU_Vector() {}
    };


    de::GPU_Vector& CreateGPUVectorRef();


    de::GPU_Vector* CreateGPUVectorPtr();


    de::GPU_Vector& CreateGPUVectorRef(const int _type, const size_t _length);


    de::GPU_Vector* CreateGPUVectorPtr(const int _type, const size_t _length);
}

#endif