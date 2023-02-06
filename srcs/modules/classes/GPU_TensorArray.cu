/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "GPU_TensorArray.h"



void decx::_GPU_TensorArray::_attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;

    uint _alignment = 0;
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
    default:
        break;
    }

    this->dpitch = decx::utils::ceil<uint>(_depth, _alignment) * _alignment;

    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * this->_single_element_size;

    this->_init = true;
}


uint decx::_GPU_TensorArray::Width()
{
    return this->width;
}


uint decx::_GPU_TensorArray::Height()
{
    return this->height;
}


uint decx::_GPU_TensorArray::Depth()
{
    return this->depth;
}


uint decx::_GPU_TensorArray::TensorNum()
{
    return this->tensor_num;
}


int decx::_GPU_TensorArray::Type()
{
    return this->type;
}


void decx::_GPU_TensorArray::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->TensArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to allocate memory for GPU_TensorArray on device\n");
        exit(-1);
    }

    //checkCudaErrors(cudaMemset(this->TensArr.ptr, 0, this->total_bytes));
    
    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->TensptrArr, this->tensor_num * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;

    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = (uint8_t*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_single_element_size;
    }
}



void decx::_GPU_TensorArray::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->TensArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to re-allocate memory for GPU_TensorArray on device\n");
        exit(-1);
    }

    //memset(this->TensArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_realloc<void*>(&this->TensptrArr, this->tensor_num * sizeof(void*))) {
        Print_Error_Message(4, "Fail to re-allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = (uint8_t*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_single_element_size;
    }
}



void decx::_GPU_TensorArray::construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}



void decx::_GPU_TensorArray::re_construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    if (this->width != _width || this->height != _height || this->depth != _depth || this->tensor_num != _tensor_num) {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

        if (this->total_bytes > pre_size) {
            this->re_alloc_data_space();
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




void decx::_GPU_TensorArray::Load_from_host(de::TensorArray& src)
{
    _TensorArray* _src = dynamic_cast<decx::_TensorArray*>(&src);
    
    checkCudaErrors(cudaMemcpy(this->TensArr.ptr, _src->TensArr.ptr, this->total_bytes, cudaMemcpyHostToDevice));
}




void decx::_GPU_TensorArray::Load_to_host(de::TensorArray& src)
{
    _TensorArray* _src = dynamic_cast<decx::_TensorArray*>(&src);

    checkCudaErrors(cudaMemcpy(_src->TensArr.ptr, this->TensArr.ptr, this->total_bytes, cudaMemcpyDeviceToHost));
}




de::GPU_TensorArray& decx::_GPU_TensorArray::operator=(de::GPU_TensorArray& src)
{
    decx::_GPU_TensorArray& ref_src = dynamic_cast<decx::_GPU_TensorArray&>(src);

    this->_attribute_assign(ref_src.type, ref_src.width, ref_src.height, ref_src.depth, ref_src.tensor_num);

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