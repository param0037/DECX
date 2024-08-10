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


#include "../../../common/Classes/Vector.h"



void decx::_Vector::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, size_t len)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);
    uint32_t _alignment = 1;
    
    switch (this->_single_element_size)
    {
    case _SIZE_INT32_:
        _alignment = _VECTOR_ALIGN_4B_;     break;
    case _SIZE_FLOAT64_:
        _alignment = _VECTOR_ALIGN_8B_;     break;
    case _SIZE_FLOAT16_:
        _alignment = _VECTOR_ALIGN_2B_;     break;
    case _SIZE_UINT8_:
        _alignment = _VECTOR_ALIGN_1B_;     break;
    case _SIZE_VECTOR4_FP32_:
        _alignment = _VECTOR_ALIGN_16B_;    break;
    default:
        break;
    }

    this->length = len;
    this->_init = true;
    this->_length = decx::utils::ceil<size_t>(len, _alignment) * _alignment;
    this->total_bytes = this->_length * this->_single_element_size;
}



void decx::_Vector::alloc_data_space()
{
    if (decx::alloc::_host_virtual_page_malloc<void>(&this->Vec, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Vector malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }
}



void decx::_Vector::re_alloc_data_space(const uint32_t _pre_store_type)
{
    if (decx::alloc::_host_virtual_page_realloc(&this->Vec, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Vector malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }
}


void decx::_Vector::construct(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    this->_attribute_assign(_type, length);

    this->alloc_data_space();
}



void decx::_Vector::re_construct(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != _type || this->length != length)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, length);

        if (this->total_bytes > pre_size) {
            // deallocate according to the current memory pool first
            decx::alloc::_host_virtual_page_dealloc(&this->Vec);
            this->alloc_data_space();
        }
    }
}


decx::_Vector::_Vector()
{
    this->_exp_data_ptr = &this->Vec.ptr;
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0);
    this->_init = false;
}


decx::_Vector::_Vector(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    this->_exp_data_ptr = &this->Vec.ptr;
    this->_attribute_assign(_type, length);

    this->alloc_data_space();
}



size_t decx::_Vector::Len() const
{
    return this->length;
}

//
//float* decx::_Vector::ptr_fp32(size_t index)
//{
//    float* _ptr = reinterpret_cast<float*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//int* decx::_Vector::ptr_int32(size_t index)
//{
//    int* _ptr = reinterpret_cast<int*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//
//uint64_t* decx::_Vector::ptr_uint64(size_t index)
//{
//    uint64_t* _ptr = reinterpret_cast<uint64_t*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//
//double* decx::_Vector::ptr_fp64(size_t index)
//{
//    double* _ptr = reinterpret_cast<double*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//de::Half* decx::_Vector::ptr_fp16(size_t index)
//{
//    de::Half* _ptr = reinterpret_cast<de::Half*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//de::CPf* decx::_Vector::ptr_cpl32(size_t index)
//{
//    de::CPf* _ptr = reinterpret_cast<de::CPf*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//
//de::CPd* decx::_Vector::ptr_cpl64(size_t index)
//{
//    de::CPd* _ptr = reinterpret_cast<de::CPd*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//
//uint8_t* decx::_Vector::ptr_uint8(size_t index)
//{
//    uint8_t* _ptr = reinterpret_cast<uint8_t*>(this->Vec.ptr);
//    return _ptr + index;
//}
//
//
//de::Vector4f* decx::_Vector::ptr_vec4f(size_t index)
//{
//    de::Vector4f* _ptr = reinterpret_cast<de::Vector4f*>(this->Vec.ptr);
//    return _ptr + index;
//}


#if _CPP_EXPORT_ENABLED_
de::Vector& de::CreateVectorRef()
{
    return *(new decx::_Vector());
}


de::Vector* de::CreateVectorPtr()
{
    return new decx::_Vector();
}


de::Vector& de::CreateVectorRef(const de::_DATA_TYPES_FLAGS_ _type, size_t len)
{
    return *(new decx::_Vector(_type, len));
}


de::Vector* de::CreateVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, size_t len)
{
    return new decx::_Vector(_type, len);
}
#endif


void decx::_Vector::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->Vec);
}



de::Vector& decx::_Vector::SoftCopy(de::Vector& src)
{
    const decx::_Vector& ref_src = dynamic_cast<decx::_Vector&>(src);

    this->_attribute_assign(ref_src.type, ref_src.length);

    decx::alloc::_host_virtual_page_malloc_same_place(&this->Vec);

    return *this;
}



decx::_Vector::~_Vector()
{
    this->release();
}


de::_DATA_TYPES_FLAGS_ decx::_Vector::Type() const
{
    return this->type;
}


void decx::_Vector::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}


bool decx::_Vector::is_init() const
{
    return this->_init;
}


uint64_t decx::_Vector::_Length() const
{
    return this->_length;
}


uint64_t decx::_Vector::get_total_bytes() const
{
    return this->total_bytes;
}



#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    _DECX_API_ DECX_Vector DE_CreateEmptyVector()
    {
        return (DECX_Vector)(new decx::_Vector());
    }


    _DECX_API_ DECX_Vector DE_CreateVector(const int8_t type, const uint32_t len)
    {
        return (DECX_Vector)(new decx::_Vector(static_cast<de::_DATA_TYPES_FLAGS_>(type), len));
    }


    _DECX_API_ uint64_t DE_GetVectorLength(DECX_Vector src)
    {
        return ((decx::_Vector*)src)->Len();
    }
#ifdef __cplusplus
}
#endif
#endif