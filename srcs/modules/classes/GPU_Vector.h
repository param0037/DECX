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


        virtual size_t Len() const = 0;



        virtual void release() = 0;


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src) = 0;


        virtual int Type() const = 0;


        ~GPU_Vector() {}
    };
}




namespace decx
{
    class _DECX_API_ _GPU_Vector : public de::GPU_Vector
    {
    private:
        bool _init;

    public:
        size_t length,
            _length,    // It is aligned with 4
            total_bytes;

        int type, _single_element_size;


        decx::PtrInfo<void> Vec;


        _GPU_Vector();


        void _attribute_assign(const int _type, size_t length);


        void construct(const int _type, const size_t length);


        void re_construct(const int _type, const size_t length);


        void alloc_data_space();


        _GPU_Vector(const int _type, size_t length);


        virtual uint64_t Len() const;


        virtual void release();


        virtual de::GPU_Vector& SoftCopy(de::GPU_Vector& src);


        virtual int Type() const;


        bool is_init() const;


        uint64_t _Length() const;


        uint64_t get_total_bytes() const;


        ~_GPU_Vector();
    };
}




#endif