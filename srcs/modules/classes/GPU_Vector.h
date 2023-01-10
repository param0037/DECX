/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _GPU_VECTOR_H_
#define _GPU_VECTOR_H_

#include "../core/basic.h"
#include "../core/allocators.h"
#include "Vector.h"

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


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src) = 0;


        virtual int Type() = 0;


        ~GPU_Vector() {}
    };
}




namespace decx
{
    class _DECX_API_ _GPU_Vector : public de::GPU_Vector
    {
    public:
        size_t length,
            _length,    // It is aligned with 4
            total_bytes;

        int type, _single_element_size;

        decx::PtrInfo<void> Vec;


        _GPU_Vector();


        void _attribute_assign(const int _type, size_t length);


        _GPU_Vector(const int _type, size_t length);


        virtual size_t Len() { return this->length; }


        virtual void load_from_host(de::Vector& src);


        virtual void load_to_host(de::Vector& dst);

        
        virtual void release();


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src);


        virtual int Type();


        ~_GPU_Vector() {}
    };
}




#endif