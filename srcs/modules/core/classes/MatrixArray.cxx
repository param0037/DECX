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


#include "../../../common/Classes/MatrixArray.h"


void decx::_MatrixArray::alloc_data_space()
{
    for (uint32_t i = 0; i < this->_matrix_number; ++i){
        const auto* p_layout = this->_layouts[i];
        this->MatptrArr.emplace_back();

        const uint64_t alloc_bytes = (uint64_t)p_layout->height * (uint64_t)p_layout->pitch;
        if (decx::alloc::_host_virtual_page_malloc<void>(this->MatptrArr[i], alloc_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
    }
}


void decx::_MatrixArray::re_alloc_data_space()
{
    this->MatptrArr.clear();

    for (uint32_t i = 0; i < this->_matrix_number; ++i){
        const auto* p_layout = this->_layouts[i];
        this->MatptrArr.emplace_back();

        const uint64_t alloc_bytes = (uint64_t)p_layout->height * (uint64_t)p_layout->pitch;
        if (decx::alloc::_host_virtual_page_malloc<void>(this->MatptrArr[i], alloc_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
    }
}


void decx::_MatrixArray::construct(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint _MatrixNum)
{
    this->_attribute_assign(_type, _width, _height, _MatrixNum);

    this->alloc_data_space();
}



void decx::_MatrixArray::re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, const uint32_t matrix_id)
{
    
}



void decx::_MatrixArray::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint MatrixNum)
{
    this->type = _type;

    uint64_t _total_alloc_bytes = 0;

    for (int32_t i = 0; i < MatrixNum; ++i){
        this->_layouts.emplace_back();
        auto* layout_ptr = this->_layouts.back();
        layout_ptr->_attribute_assign(_type, _width, _height);

        uint64_t _mat_size = (uint64_t)layout_ptr->width * (uint64_t)layout_ptr->height;
        _total_alloc_bytes += _mat_size * layout_ptr->_single_element_size;
    }

    this->_single_element_size = this->_layouts.front()->_single_element_size;
}



decx::_MatrixArray::_MatrixArray()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
    this->_init = false;
}



decx::_MatrixArray::_MatrixArray(const de::_DATA_TYPES_FLAGS_ _type, uint W, uint H, uint MatrixNum)
{
    this->_attribute_assign(_type, W, H, MatrixNum);

    this->alloc_data_space();
}


uint32_t decx::_MatrixArray::Width(const uint32_t matrix_id) const { return this->_layouts[matrix_id]->height; }


uint32_t decx::_MatrixArray::Height(const uint32_t matrix_id) const { return this->_layouts[matrix_id]->height; }


uint32_t decx::_MatrixArray::MatrixNumber() const { return this->ArrayNumber; }


#if _CPP_EXPORT_ENABLED_
de::MatrixArray& de::CreateMatrixArrayRef()
{
    return *(new decx::_MatrixArray());
}


de::MatrixArray* de::CreateMatrixArrayPtr()
{
    return new decx::_MatrixArray();
}


de::MatrixArray& de::CreateMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum)
{
    return *(new decx::_MatrixArray(_type, width, height, MatrixNum));
}


de::MatrixArray* de::CreateMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum)
{
    return new decx::_MatrixArray(_type, width, height, MatrixNum);
}
#endif


void decx::_MatrixArray::release()
{
    for (int32_t i = 0; i < this->_matrix_number; ++i) {
        decx::alloc::_host_virtual_page_dealloc(this->MatptrArr[i]);
    }
    
    this->_layouts.~Dynamic_Array();
    this->MatptrArr.~Dynamic_Array();
}



de::_DATA_TYPES_FLAGS_ decx::_MatrixArray::Type() const
{
    return this->type;
}


void decx::_MatrixArray::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}


de::MatrixArray& decx::_MatrixArray::SoftCopy(de::MatrixArray& src)
{
    const decx::_MatrixArray& ref_src = dynamic_cast<decx::_MatrixArray&>(src);

    this->_layouts = ref_src._layouts;

    for (int32_t i = 0; i < _matrix_number; ++i){
        decx::alloc::_host_virtual_page_malloc_same_place(this->MatptrArr[i]);
    }

    return *this;
}



uint32_t decx::_MatrixArray::Pitch(const uint32_t matrix_id) const
{
    return this->_layouts[matrix_id]->pitch;
}


uint32_t decx::_MatrixArray::Array_num() const
{
    return this->ArrayNumber;
}


const decx::_matrix_layout& decx::_MatrixArray::get_layout(const uint32_t matrix_id) const
{
    return *this->_layouts[matrix_id];
}



bool decx::_MatrixArray::is_init() const
{
    return this->_init;
}



#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    _DECX_API_ DECX_MatrixArray DE_CreateEmptyMatrixArray()
    {
        return (DECX_MatrixArray)(new decx::_MatrixArray());
    }


    _DECX_API_ DECX_MatrixArray DE_CreateMatrixArray(const int8_t type, const uint32_t width, const uint32_t height,
        uint32_t MatrixNum)
    {
        return (DECX_MatrixArray)(new decx::_MatrixArray(static_cast<de::_DATA_TYPES_FLAGS_>(type), width, height, MatrixNum));
    }


    /*_DECX_API_ DECX_Handle DE_GetMatrixArrayProp(const DECX_MatrixArray src, DECX_MatrixLayout* prop)
    {
        decx::_MatrixArray* _src = (decx::_MatrixArray*)src;
        de::DH handle;

        if (prop == NULL) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
                INVALID_PARAM);
            return _CAST_HANDLE_(DECX_Handle, handle);
        }
        memcpy(prop, &_src->get_layout(), sizeof(DECX_MatrixLayout));
        return _CAST_HANDLE_(DECX_Handle, handle);
    }*/
#ifdef __cplusplus
}
#endif
#endif
