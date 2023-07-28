/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "GPU_TensorArray.h"



void decx::_GPU_TensorArray::_attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->type = _type;
    
    this->type = _type;
    this->tensor_num = _tensor_num;
    this->_init = true;

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->_gap = this->_layout.dp_x_wp * static_cast<size_t>(this->_layout.height);

    this->element_num = static_cast<size_t>(this->_layout.depth) * this->_layout.plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * this->_layout._single_element_size;
}


uint decx::_GPU_TensorArray::Width() const
{
    return this->_layout.width;
}


uint decx::_GPU_TensorArray::Height() const
{
    return this->_layout.height;
}


uint decx::_GPU_TensorArray::Depth() const
{
    return this->_layout.depth;
}


uint decx::_GPU_TensorArray::TensorNum() const
{
    return this->tensor_num;
}


int decx::_GPU_TensorArray::Type() const
{
    return this->type;
}



const decx::_tensor_layout& decx::_GPU_TensorArray::get_layout() const
{
    return this->_layout;
}


void decx::_GPU_TensorArray::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->TensArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to allocate memory for GPU_TensorArray on device\n");
        exit(-1);
    }

    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->TensptrArr, this->tensor_num * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;

    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = (uint8_t*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_layout._single_element_size;
    }
}



void decx::_GPU_TensorArray::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->TensArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to re-allocate memory for GPU_TensorArray on device\n");
        exit(-1);
    }

    if (decx::alloc::_host_virtual_page_realloc<void*>(&this->TensptrArr, this->tensor_num * sizeof(void*))) {
        Print_Error_Message(4, "Fail to re-allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = (uint8_t*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_layout._single_element_size;
    }
}



void decx::_GPU_TensorArray::construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}



void decx::_GPU_TensorArray::re_construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    if (this->type != _type || this->_layout.width != _width || this->_layout.height != _height || 
        this->_layout.depth != _depth || this->tensor_num != _tensor_num)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

        if (this->total_bytes > pre_size)
        {
            decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);

            if (this->TensArr.ptr != NULL) {
                decx::alloc::_device_dealloc(&this->TensArr);

                this->alloc_data_space();
            }
            else {
                this->re_alloc_data_space();
            }
        }
        else {
            this->TensptrArr.ptr[0] = this->TensArr.ptr;
            for (uint i = 1; i < this->tensor_num; ++i) {
                this->TensptrArr.ptr[i] = (void*)((uchar*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_layout._single_element_size);
            }
        }
    }
}


decx::_GPU_TensorArray::_GPU_TensorArray()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0);

    this->_init = false;
}



decx::_GPU_TensorArray::_GPU_TensorArray(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}




de::GPU_TensorArray& decx::_GPU_TensorArray::SoftCopy(const de::GPU_TensorArray& src)
{
    const decx::_GPU_TensorArray& ref_src = dynamic_cast<const decx::_GPU_TensorArray&>(src);

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth, ref_src.tensor_num);

    decx::alloc::_device_malloc_same_place(&this->TensArr);
    
    memset(this->TensArr.ptr, 0, this->total_bytes);

    decx::alloc::_host_virtual_page_malloc_same_place<void*>(&this->TensptrArr);

    return *this;
}




void decx::_GPU_TensorArray::release()
{
    decx::alloc::_device_dealloc(&this->TensArr);

    decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);
}



bool decx::_GPU_TensorArray::is_init() const
{
    return this->_init;
}


uint64_t decx::_GPU_TensorArray::get_total_bytes() const
{
    return this->total_bytes;
}



namespace de
{
    
    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef();


    
    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr();


    
    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num);


    
    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num);
}


_DECX_API_
de::GPU_TensorArray& de::CreateGPUTensorArrayRef()
{
    return *(new decx::_GPU_TensorArray());
}



_DECX_API_
de::GPU_TensorArray* de::CreateGPUTensorArrayPtr()
{
    return new decx::_GPU_TensorArray();
}



_DECX_API_
de::GPU_TensorArray& de::CreateGPUTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num)
{
    return *(new decx::_GPU_TensorArray(_type, width, height, depth, tensor_num));
}



_DECX_API_
de::GPU_TensorArray* de::CreateGPUTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num)
{
    return new decx::_GPU_TensorArray(_type, width, height, depth, tensor_num);
}