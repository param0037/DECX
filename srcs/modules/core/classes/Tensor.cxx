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


#include "../../../common/Classes/Tensor.h"


void decx::_tensor_layout::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width,
    const uint32_t _height, const uint32_t _depth)
{
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->wpitch = decx::utils::align<uint32_t>(_width, 4);

    uint32_t _alignment = 1;
    switch (this->_single_element_size)
    {
    case 4:
        _alignment = _TENSOR_ALIGN_DEPTH_4B_;     break;
    case 8:
        _alignment = _TENSOR_ALIGN_DEPTH_8B_;     break;
    case 2:
        _alignment = _TENSOR_ALIGN_DEPTH_2B_;     break;
    case 1:
        _alignment = _TENSOR_ALIGN_DEPTH_1B_;     break;
    case 16:
        _alignment = _TENSOR_ALIGN_DEPTH_16B_;    break;
    default:
        break;
    }
    this->dpitch = decx::utils::align<uint32_t>(_depth, _alignment);

    this->dp_x_wp = static_cast<uint64_t>(this->dpitch) * static_cast<uint64_t>(this->wpitch);

    this->plane[0] = static_cast<uint64_t>(this->height) * static_cast<uint64_t>(this->width);
    this->plane[1] = static_cast<uint64_t>(this->depth) * static_cast<uint64_t>(this->width);
    this->plane[2] = static_cast<uint64_t>(this->height) * static_cast<uint64_t>(this->depth);
}



decx::_Tensor::_Tensor()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
    this->_exp_data_ptr = &this->Tens.ptr;
    this->_exp_tensor_dscr = &this->_layout;

    this->_init = false;
}



void decx::_Tensor::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    this->type = _type;
    
    this->_init = true;

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->element_num = static_cast<uint64_t>(this->_layout.depth) * this->_layout.plane[0];
    this->_element_num = static_cast<uint64_t>(this->_layout.height) * this->_layout.dp_x_wp;
    this->total_bytes = this->_element_num * this->_layout._single_element_size;
}



void decx::_Tensor::alloc_data_space()
{
    if (decx::alloc::_host_virtual_page_malloc<void>(&this->Tens, this->total_bytes)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your RAM.");
        exit(-1);
    }

    memset(this->Tens.ptr, 0, this->total_bytes);
}



void decx::_Tensor::re_alloc_data_space()
{
    decx::alloc::_host_virtual_page_realloc<void>(&this->Tens, this->total_bytes);

    memset(this->Tens.ptr, 0, this->total_bytes);
}



void decx::_Tensor::construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}



void decx::_Tensor::re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != _type || this->_layout.width != _width || this->_layout.height != _height)
    {
        const uint64_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth);

        if (this->total_bytes > pre_size) {
            // deallocate according to the current memory pool first
            decx::alloc::_host_virtual_page_dealloc(&this->Tens);
            this->alloc_data_space();
        }
    }
}



decx::_Tensor::_Tensor(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    this->_exp_data_ptr = &this->Tens.ptr;
    this->_exp_tensor_dscr = &this->_layout;

    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}




void decx::_Tensor::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->Tens);
}



de::_DATA_TYPES_FLAGS_ decx::_Tensor::Type() const
{
    return this->type;
}


void decx::_Tensor::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}



const decx::_tensor_layout& decx::_Tensor::get_layout() const
{
    return this->_layout;
}


bool decx::_Tensor::is_init() const
{
    return this->_init;
}


uint64_t decx::_Tensor::get_total_bytes() const
{
    return this->total_bytes;
}


decx::_tensor_layout& decx::_Tensor::get_layout_modify()
{
    return this->_layout;
}

#if _CPP_EXPORT_ENABLED_
de::Tensor& de::CreateTensorRef()
{
    return *(new decx::_Tensor());
}


de::Tensor* de::CreateTensorPtr()
{
    return new decx::_Tensor();
}


de::Tensor& de::CreateTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    return *(new decx::_Tensor(_type, _width, _height, _depth));
}


de::Tensor* de::CreateTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    return new decx::_Tensor(_type, _width, _height, _depth);
}
#endif


uint32_t decx::_Tensor::Width() const { return this->_layout.width; }
uint32_t decx::_Tensor::Height() const { return this->_layout.height; }
uint32_t decx::_Tensor::Depth() const { return this->_layout.depth; }


de::Tensor& decx::_Tensor::SoftCopy(de::Tensor& src)
{
    decx::_Tensor& ref_src = dynamic_cast<decx::_Tensor&>(src);

    this->Tens.block = ref_src.Tens.block;

    this->_attribute_assign(ref_src.Type(), ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth);

    decx::alloc::_host_virtual_page_malloc_same_place(&this->Tens);

    return *this;
}



#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    _DECX_API_ DECX_Tensor DE_CreateEmptyTensor()
    {
        return (DECX_Tensor)(new decx::_Tensor());
    }


    _DECX_API_ DECX_Tensor DE_CreateTensor(const int8_t type, const uint32_t _width, const uint32_t _height,
        const uint32_t _depth)
    {
        return (DECX_Tensor)(new decx::_Tensor(static_cast<de::_DATA_TYPES_FLAGS_>(type), _width, _height, _depth));
    }

    
    _DECX_API_ DECX_Handle DE_GetTensorProp(const DECX_Tensor src, DECX_TensorLayout* prop)
    {
        decx::_Tensor* _src = (decx::_Tensor*)src;
        de::DH handle;

        if (prop == NULL) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_INVALID_PARAM,
                INVALID_PARAM);
            return _CAST_HANDLE_(DECX_Handle, handle);
        }
        memcpy(prop, &_src->get_layout(), sizeof(DECX_TensorLayout));
        return _CAST_HANDLE_(DECX_Handle, handle);
    }
#ifdef __cplusplus
}
#endif
#endif
