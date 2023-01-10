/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#include "GPU_Tensor.h"



void decx::_GPU_Tensor::_attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;

    uint _alignment = 0;
    switch (this->_single_element_size)
    {
    case 4:
        _alignment = _TENSOR_ALIGN_4B_;     break;
    case 8:
        _alignment = _TENSOR_ALIGN_8B_;     break;
    case 2:
        _alignment = _TENSOR_ALIGN_2B_;     break;
    case 1:
        _alignment = _TENSOR_ALIGN_1B_;     break;
    default:
        break;
    }
    this->dpitch = decx::utils::ceil<uint>(_depth, _alignment) * _alignment;

    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * this->_single_element_size;
}




void decx::_GPU_Tensor::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->Tens, this->total_bytes)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your device.");
        exit(-1);
    }
    checkCudaErrors(cudaMemset(this->Tens.ptr, 0, this->total_bytes));
}




void decx::_GPU_Tensor::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->Tens, this->total_bytes)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your device.");
        exit(-1);
    }
    checkCudaErrors(cudaMemset(this->Tens.ptr, 0, this->total_bytes));
}



void decx::_GPU_Tensor::construct(const int _type, const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}




void decx::_GPU_Tensor::re_construct(const int _type, const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->re_alloc_data_space();
}




decx::_GPU_Tensor::_GPU_Tensor(const int _type, const uint _width, const uint _height, const uint _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}




decx::_GPU_Tensor::_GPU_Tensor()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
}




de::GPU_Tensor& decx::_GPU_Tensor::SoftCopy(de::GPU_Tensor& src)
{
    decx::_GPU_Tensor& ref_src = dynamic_cast<decx::_GPU_Tensor &>(src);

    this->Tens.block = ref_src.Tens.block;

    this->_attribute_assign(ref_src.type, ref_src.width, ref_src.height, ref_src.depth);

    decx::alloc::_device_malloc_same_place(&this->Tens);

    return *this;
}




void decx::_GPU_Tensor::Load_from_host(de::Tensor& src)
{
    decx::_Tensor& ref_src = dynamic_cast<decx::_Tensor&>(src);
    if (ref_src.type != this->type) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        return;
    }

    checkCudaErrors(cudaMemcpy(this->Tens.ptr, ref_src.Tens.ptr, this->total_bytes, cudaMemcpyHostToDevice));
}




void decx::_GPU_Tensor::Load_to_host(de::Tensor& dst)
{
    decx::_Tensor& ref_dst = dynamic_cast<decx::_Tensor&>(dst);
    if (ref_dst.type != this->type) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        return;
    }

    checkCudaErrors(cudaMemcpy(ref_dst.Tens.ptr, this->Tens.ptr, this->total_bytes, cudaMemcpyDeviceToHost));
}




void decx::_GPU_Tensor::release()
{
    decx::alloc::_device_dealloc(&this->Tens);
}


namespace de
{
    de::GPU_Tensor* CreateGPUTensorPtr();


    de::GPU_Tensor& CreateGPUTensorRef();


    de::GPU_Tensor* CreateGPUTensorPtr(const int _type, const uint _width, const uint _height, const uint _depth);


    de::GPU_Tensor& CreateGPUTensorRef(const int _type, const uint _width, const uint _height, const uint _depth);
}




de::GPU_Tensor& de::CreateGPUTensorRef()
{
    return *(new decx::_GPU_Tensor());
}



de::GPU_Tensor* de::CreateGPUTensorPtr()
{
    return new decx::_GPU_Tensor();
}



de::GPU_Tensor& de::CreateGPUTensorRef(const int _type, const uint _width, const uint _height, const uint _depth)
{
    return *(new decx::_GPU_Tensor(_type, _width, _height, _depth));
}



de::GPU_Tensor* de::CreateGPUTensorPtr(const int _type, const uint _width, const uint _height, const uint _depth)
{
    return new decx::_GPU_Tensor(_type, _width, _height, _depth);
}


int decx::_GPU_Tensor::Type()
{
    return this->type;
}