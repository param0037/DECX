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

#ifdef __cplusplus
namespace de
{
    class _DECX_API_ Vector
    {
    protected:
        _SHADOW_ATTRIBUTE_(void*) _exp_data_ptr;

    public:
        Vector() {}


        virtual size_t Len() const = 0;


        template <typename _ptr_type>
        _ptr_type* ptr(const uint64_t _idx)
        {
            return ((_ptr_type*)*this->_exp_data_ptr) + _idx;
        }


        virtual void release() = 0;


        virtual de::Vector& SoftCopy(de::Vector& src) = 0;

        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        ~Vector() {}
    };


    typedef const de::Vector& InputVector;
    typedef de::Vector& OutputVector;
    typedef de::Vector& InOutVector;


    _DECX_API_ de::Vector& CreateVectorRef();


    _DECX_API_ de::Vector* CreateVectorPtr();


    _DECX_API_ de::Vector& CreateVectorRef(const de::_DATA_TYPES_FLAGS_ _type, size_t len);


    _DECX_API_ de::Vector* CreateVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, size_t len);
}
#endif


#ifdef _C_CONTEXT_
typedef struct DECX_Vector_t
{
    void* _segment;
}DECX_Vector;


_DECX_API_ DECX_Vector DE_CreateEmptyVector();


_DECX_API_ DECX_Vector DE_CreateVector(const int8_t type, const uint32_t len);
#endif


#endif