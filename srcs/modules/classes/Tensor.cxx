/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Tensor.h"


void decx::_tensor_layout::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width,
    const uint _height, const uint _depth)
{
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;

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
    this->dpitch = decx::utils::ceil<uint>(_depth, _alignment) * _alignment;

    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);
}



decx::_Tensor::_Tensor()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
    this->_init = false;
}



void decx::_Tensor::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth)
{
    this->type = _type;
    
    this->_init = true;

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->element_num = static_cast<size_t>(this->_layout.depth) * this->_layout.plane[0];
    this->_element_num = static_cast<size_t>(this->_layout.height) * this->_layout.dp_x_wp;
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



void decx::_Tensor::construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}




void decx::_Tensor::re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != _type || this->_layout.width != _width || this->_layout.height != _height)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth);

        if (this->total_bytes > pre_size) {
            // deallocate according to the current memory pool first
            decx::alloc::_host_virtual_page_dealloc(&this->Tens);
            this->alloc_data_space();
        }
    }
}




decx::_Tensor::_Tensor(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth)
{
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



float* decx::_Tensor::ptr_fp32(const int x, const int y, const int z)
{
    float* ptr = reinterpret_cast<float*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
}


int* decx::_Tensor::ptr_int32(const int x, const int y, const int z)
{
    int* ptr = reinterpret_cast<int*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
}


uint8_t* decx::_Tensor::ptr_uint8(const int x, const int y, const int z)
{
    uint8_t* ptr = reinterpret_cast<uint8_t*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
}


de::CPf* decx::_Tensor::ptr_cpl32(const int x, const int y, const int z)
{
    de::CPf* ptr = reinterpret_cast<de::CPf*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
}


double* decx::_Tensor::ptr_fp64(const int x, const int y, const int z)
{
    double* ptr = reinterpret_cast<double*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
}


de::Half* decx::_Tensor::ptr_fp16(const int x, const int y, const int z)
{
    de::Half* ptr = reinterpret_cast<de::Half*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
}


de::Vector4f* decx::_Tensor::ptr_vec4f(const int x, const int y, const int z)
{
    de::Vector4f* ptr = reinterpret_cast<de::Vector4f*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z);
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


_DECX_API_ de::Tensor& de::CreateTensorRef()
{
    return *(new decx::_Tensor());
}



_DECX_API_ de::Tensor* de::CreateTensorPtr()
{
    return new decx::_Tensor();
}



_DECX_API_ de::Tensor& de::CreateTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    return *(new decx::_Tensor(_type, _width, _height, _depth));
}



_DECX_API_ de::Tensor* de::CreateTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    return new decx::_Tensor(_type, _width, _height, _depth);
}



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
