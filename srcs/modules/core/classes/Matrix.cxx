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


#include "../../../common/Classes/Matrix.h"


void decx::_matrix_layout::
_attribute_assign(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height)
{
    this->_single_element_size = decx::core::_size_mapping(type);

    this->width = _width;
    this->height = _height;

    uint32_t _alignment = 1;

    if (type != de::_DATA_TYPES_FLAGS_::_VOID_)
    {
        switch (this->_single_element_size)
        {
        case 4:
            _alignment = _MATRIX_ALIGN_4B_;     break;
        case 16:
            _alignment = _MATRIX_ALIGN_16B_;     break;
        case 8:
            _alignment = _MATRIX_ALIGN_8B_;     break;
        case 2:
            _alignment = _MATRIX_ALIGN_2B_;     break;
        case 1:
            _alignment = _MATRIX_ALIGN_1B_;     break;
        default:
            break;
        }
    }
    this->pitch = decx::utils::align<uint>(_width, _alignment);
}




void decx::_Matrix::alloc_data_space()
{
    if (decx::alloc::_host_virtual_page_malloc<void>(&this->Mat, this->total_bytes)) {
        
        return;
    }
}



void decx::_Matrix::re_alloc_data_space()
{
    if (decx::alloc::_host_virtual_page_realloc<void>(&this->Mat, this->total_bytes)) {
        
        return;
    }
}


de::_DATA_TYPES_FLAGS_ decx::_Matrix::Type() const
{
    return this->type;
}


void decx::_Matrix::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}


de::_DATA_FORMATS_ decx::_Matrix::Format() const
{
    return this->_format;
}


uint32_t decx::_Matrix::Pitch() const
{
    return this->_layout.pitch;
}


const decx::_matrix_layout& decx::_Matrix::get_layout() const
{
    return this->_layout;
}


void decx::_Matrix::construct(const de::_DATA_TYPES_FLAGS_ type, uint32_t _width, uint32_t _height,
    const de::_DATA_FORMATS_ format)
{
    this->_attribute_assign(type, _width, _height);
    this->_format = format;

    this->alloc_data_space();
}



void decx::_Matrix::re_construct(const de::_DATA_TYPES_FLAGS_ type, uint32_t _width, uint32_t _height)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->_layout.width != _width || this->_layout.height != _height)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(type, _width, _height);

        if (this->total_bytes > pre_size) {
            // deallocate according to the current memory pool first
            decx::alloc::_host_virtual_page_dealloc(&this->Mat);
            this->alloc_data_space();
        }
    }
}



void decx::_Matrix::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height)
{
    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);

    this->_init = true;

    this->_init = _type != de::_DATA_TYPES_FLAGS_::_VOID_;
    uint64_t element_num = static_cast<size_t>(this->_layout.pitch) * static_cast<size_t>(_height);
    this->total_bytes = (element_num)*this->_layout._single_element_size;
}




void decx::_Matrix::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
}


decx::_Matrix::_Matrix()
{
    this->_exp_data_ptr = &this->Mat.ptr;
    this->_matrix_dscr = &this->_layout;
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0);
    this->_init = false;
}



decx::_Matrix::_Matrix(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
    de::_DATA_FORMATS_ format)
{
    this->_exp_data_ptr = &this->Mat.ptr;
    this->_matrix_dscr = &this->_layout;
    this->construct(type, _width, _height, format);
}


uint32_t decx::_Matrix::Width() const
{
    return this->_layout.width;
}


uint32_t decx::_Matrix::Height() const
{
    return this->_layout.height;
}

decx::_Matrix::~_Matrix()
{
    if (this->Mat.ptr != NULL) {
        this->release();
    }
}


bool decx::_Matrix::is_init() const
{
    return this->_init;
}


uint64_t decx::_Matrix::get_total_bytes() const
{
    return this->total_bytes;
}


de::_DATA_FORMATS_ decx::_Matrix::get_data_format() const
{
    return this->_format;
}


void decx::_Matrix::set_data_format(const de::_DATA_FORMATS_& format)
{
    this->_format = format;
}


de::Matrix& decx::_Matrix::SoftCopy(de::Matrix& src)
{
    const decx::_Matrix& ref_src = dynamic_cast<decx::_Matrix&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height);

    decx::alloc::_host_virtual_page_malloc_same_place(&this->Mat);

    return *this;
}


#if _CPP_EXPORT_ENABLED_
de::Matrix& de::CreateMatrixRef()
{
    return *(new decx::_Matrix());
}


de::Matrix* de::CreateMatrixPtr()
{
    return new decx::_Matrix();
}


de::Matrix& de::CreateMatrixRef(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
    const de::_DATA_FORMATS_ format)
{
    return *(new decx::_Matrix(type, _width, _height, format));
}


de::Matrix* de::CreateMatrixPtr(const de::_DATA_TYPES_FLAGS_ type, const uint32_t _width, const uint32_t _height,
    const de::_DATA_FORMATS_ format)
{
    return new decx::_Matrix(type, _width, _height, format);
}
#endif  // #if _CPP_EXPORT_ENABLED_


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    _DECX_API_ DECX_Matrix DE_CreateEmptyMatrix()
    {
        return (DECX_Matrix)(new decx::_Matrix());
    }


    _DECX_API_ DECX_Matrix DE_CreateMatrix(const int8_t type, const uint32_t _width, const uint32_t _height,
        const int8_t format)
    {
        return (DECX_Matrix)(new decx::_Matrix(static_cast<de::_DATA_TYPES_FLAGS_>(type), _width, _height, 
            static_cast<de::_DATA_FORMATS_>(format)));
    }


    _DECX_API_ DECX_Handle DE_GetMatrixProp(const DECX_Matrix src, DECX_MatrixLayout* prop)
    {
        decx::_Matrix* _src = (decx::_Matrix*)src;
        de::DH handle;

        if (prop == NULL) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
                INVALID_PARAM);
            return _CAST_HANDLE_(DECX_Handle, handle);
        }
        memcpy(prop, &_src->get_layout(), sizeof(DECX_MatrixLayout));
        return _CAST_HANDLE_(DECX_Handle, handle);
    }
#ifdef __cplusplus
}
#endif
#endif
