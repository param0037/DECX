/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Vector.h"



void decx::_Vector::_attribute_assign(const int _type, size_t len, const int flag)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);
    uint _alignment = 0;

    this->Vec.ptr = NULL;           this->Vec.block = NULL;

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
    this->_store_type = flag;
}



void decx::_Vector::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<void>(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<void>(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;
    }
}




void decx::_Vector::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc(&this->Vec, this->total_bytes)) {
            SetConsoleColor(4);
            printf("Vector malloc failed! Please check if there is enough space in your device.");
            ResetConsoleColor;
            return;
        }
        break;
    }
}



void decx::_Vector::construct(const int _type, size_t length, const int flag)
{
    this->_attribute_assign(_type, length, flag);

    this->alloc_data_space();
}



void decx::_Vector::re_construct(const int _type, size_t length, const int flag)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->length != _length || this->_store_type != flag)
    {
        const size_t pre_size = this->total_bytes;
        const int pre_store_type = this->_store_type;

        this->_attribute_assign(_type, length, flag);

        if (this->total_bytes > pre_size || pre_store_type != flag) {
            // deallocate according to the current memory pool first
            if (pre_store_type != flag && this->Vec.ptr == NULL) {
                switch (pre_store_type)
                {
                case decx::DATA_STORE_TYPE::Page_Default:
                    decx::alloc::_host_virtual_page_dealloc(&this->Vec);
                    break;

                case decx::DATA_STORE_TYPE::Page_Locked:
                    decx::alloc::_host_fixed_page_dealloc(&this->Vec);
                    break;
                default:
                    break;
                }
                this->alloc_data_space();
            }
            else {
                this->re_alloc_data_space();
            }
        }
    }
}




decx::_Vector::_Vector()
{
    this->_attribute_assign(_VOID_, 0, 0);
    this->_init = false;
}




decx::_Vector::_Vector(const int _type, size_t length, const int flag)
{
    this->_attribute_assign(_type, length, flag);

    this->alloc_data_space();
}



size_t decx::_Vector::Len()
{
    return this->length;
}


float* decx::_Vector::ptr_fp32(size_t index)
{
    float* _ptr = reinterpret_cast<float*>(this->Vec.ptr);
    return _ptr + index;
}

int* decx::_Vector::ptr_int32(size_t index)
{
    int* _ptr = reinterpret_cast<int*>(this->Vec.ptr);
    return _ptr + index;
}

double* decx::_Vector::ptr_fp64(size_t index)
{
    double* _ptr = reinterpret_cast<double*>(this->Vec.ptr);
    return _ptr + index;
}

de::Half* decx::_Vector::ptr_fp16(size_t index)
{
    de::Half* _ptr = reinterpret_cast<de::Half*>(this->Vec.ptr);
    return _ptr + index;
}

de::CPf* decx::_Vector::ptr_cpl32(size_t index)
{
    de::CPf* _ptr = reinterpret_cast<de::CPf*>(this->Vec.ptr);
    return _ptr + index;
}

uint8_t* decx::_Vector::ptr_uint8(size_t index)
{
    uint8_t* _ptr = reinterpret_cast<uint8_t*>(this->Vec.ptr);
    return _ptr + index;
}


de::Vector4f* decx::_Vector::ptr_vec4f(size_t index)
{
    de::Vector4f* _ptr = reinterpret_cast<de::Vector4f*>(this->Vec.ptr);
    return _ptr + index;
}


namespace de
{
    _DECX_API_ de::Vector& CreateVectorRef();


    _DECX_API_ de::Vector* CreateVectorPtr();


    _DECX_API_ de::Vector& CreateVectorRef(const int _type, size_t len, const int flag);


    _DECX_API_ de::Vector* CreateVectorPtr(const int _type, size_t len, const int flag);
}



de::Vector& de::CreateVectorRef()
{
    return *(new decx::_Vector());
}



de::Vector* de::CreateVectorPtr()
{
    return new decx::_Vector();
}



de::Vector& de::CreateVectorRef(const int _type, size_t len, const int flag)
{
    return *(new decx::_Vector(_type, len, flag));
}



de::Vector* de::CreateVectorPtr(const int _type, size_t len, const int flag)
{
    return new decx::_Vector(_type, len, flag);
}



void decx::_Vector::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->Vec);
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->Vec);
        break;
    }
}



de::Vector& decx::_Vector::SoftCopy(de::Vector& src)
{
    const decx::_Vector& ref_src = dynamic_cast<decx::_Vector&>(src);

    this->_attribute_assign(ref_src.type, ref_src.length, ref_src._store_type);

    switch (ref_src._store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->Vec);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->Vec);
        break;
    default:
        break;
    }

    return *this;
}
