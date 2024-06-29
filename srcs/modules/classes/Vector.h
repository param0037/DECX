/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _VECTOR_H_
#define _VECTOR_H_


#include "../core/basic.h"
#include "../core/allocators.h"
#include "../handles/decx_handles.h"
#include "classes_util.h"
#include "type_info.h"


// 39193 2022.4.28 12:49

#define _VECTOR_ALIGN_4B_ 8        // fp32, int32
#define _VECTOR_ALIGN_8B_ 4        // fp64
#define _VECTOR_ALIGN_16B_ 2       // vector4f
#define _VECTOR_ALIGN_2B_ 16       // fp16
#define _VECTOR_ALIGN_1B_ 32       // uchar


namespace de
{
    class 
#if _CPP_EXPORT_ENABLED_
        _DECX_API_
#endif 
        Vector
    {
    protected:
        _SHADOW_ATTRIBUTE_(void*) _exp_data_ptr;

    public:
        Vector() {}


        virtual size_t Len() const = 0;


        /*virtual float*              ptr_fp32(size_t index)  = 0;
        virtual int*                ptr_int32(size_t index) = 0;
        virtual uint64_t*           ptr_uint64(size_t index) = 0;
        virtual double*             ptr_fp64(size_t index)  = 0;
        virtual de::Half*           ptr_fp16(size_t index)  = 0;
        virtual de::CPf*            ptr_cpl32(size_t index) = 0;
        virtual de::CPd*            ptr_cpl64(size_t index) = 0;
        virtual uint8_t*            ptr_uint8(size_t index) = 0;
        virtual de::Vector4f*       ptr_vec4f(size_t index) = 0;*/


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
}



/*
* Data storage structure
* 
* <---------------- _length -------------------->
* <------------- length ----------->
* [x x x x x x x x x x x x x x x x x 0 0 0 0 0 0]
*/


namespace decx
{
    class _DECX_API_ _Vector : public de::Vector
    {
    private:
        void _attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, size_t len);


        void alloc_data_space();


        void re_alloc_data_space(const uint32_t _pre_store_type);

        bool _init;

    public:
        //int _store_type;        // host locked or host virtual-paged
        size_t length,
            _length,    // It is aligned with 8
            total_bytes;

        de::_DATA_TYPES_FLAGS_ type;
        uint8_t _single_element_size;

        

        decx::PtrInfo<void> Vec;


        void construct(const de::_DATA_TYPES_FLAGS_ _type, size_t length);


        void re_construct(const de::_DATA_TYPES_FLAGS_ _type, size_t length);


        _Vector();


        _Vector(const de::_DATA_TYPES_FLAGS_ _type, size_t length);


        virtual size_t Len() const;


        /*virtual float*           ptr_fp32(size_t index);
        virtual int*             ptr_int32(size_t index);
        virtual uint64_t*        ptr_uint64(size_t index);
        virtual double*          ptr_fp64(size_t index);
        virtual de::Half*        ptr_fp16(size_t index);
        virtual de::CPf*         ptr_cpl32(size_t index);
        virtual de::CPd*         ptr_cpl64(size_t index);
        virtual uint8_t*         ptr_uint8(size_t index);
        virtual de::Vector4f*    ptr_vec4f(size_t index);*/


        virtual void release();


        virtual de::Vector& SoftCopy(de::Vector &src);


        virtual de::_DATA_TYPES_FLAGS_ Type() const;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type);


        bool is_init() const;


        uint64_t _Length() const;


        uint64_t get_total_bytes() const;


        ~_Vector();
    };
}


#if _CPP_EXPORT_ENABLED_
namespace de
{
    _DECX_API_ de::Vector& CreateVectorRef();


    _DECX_API_ de::Vector* CreateVectorPtr();


    _DECX_API_ de::Vector& CreateVectorRef(const de::_DATA_TYPES_FLAGS_ _type, size_t len);


    _DECX_API_ de::Vector* CreateVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, size_t len);
}
#endif


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct decx::_Vector* DECX_Vector;


    _DECX_API_ DECX_Vector DE_CreateEmptyVector();


    _DECX_API_ DECX_Vector DE_CreateVector(const int8_t type, const uint32_t len);


    _DECX_API_ uint64_t DE_GetVectorLength(DECX_Vector src);
#ifdef __cplusplus
}
#endif      // # ifdef __cplusplus
#endif      // #if _C_EXPORT_ENABLED_



#endif
