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


#include "../../../common/Classes/GPU_Tensor.h"


void decx::_tensor_layout::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width,
    const uint32_t _height, const uint32_t _depth)
{
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    uint32_t _alignment = 1;
    uint32_t _alignment_W = 8;

    /*switch (this->_single_element_size)
    {
    case 4:
        _alignment = 32;     break;
    case 8:
        _alignment = 16;     break;
    case 2:
        _alignment = 64;     break;
    case 1:
        _alignment = 128;     break;
    case 16:
        _alignment = 8;    break;
    default:
        break;
    }*/

    switch (this->_single_element_size)
    {
    case 4:
        _alignment = 4;     break;
    case 8:
        _alignment = 2;     break;
    case 2:
        _alignment = 8;     break;
    case 1:
        _alignment = 16;     break;
    case 16:
        _alignment = 1;    break;
    default:
        break;
    }

    this->wpitch = decx::utils::ceil<uint32_t>(_width, _alignment_W) * _alignment_W;
    this->dpitch = decx::utils::ceil<uint32_t>(_depth, _alignment) * _alignment;

    this->dp_x_wp = static_cast<uint64_t>(this->dpitch) * static_cast<uint64_t>(this->wpitch);

    this->plane[0] = static_cast<uint64_t>(this->height) * static_cast<uint64_t>(this->width);
    this->plane[1] = static_cast<uint64_t>(this->depth) * static_cast<uint64_t>(this->width);
    this->plane[2] = static_cast<uint64_t>(this->height) * static_cast<uint64_t>(this->depth);
}



uint32_t decx::_GPU_Tensor::Width() const { return this->_layout.width; }
uint32_t decx::_GPU_Tensor::Height() const { return this->_layout.height; }
uint32_t decx::_GPU_Tensor::Depth() const { return this->_layout.depth; }



void decx::_GPU_Tensor::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    this->type = _type;

    this->_init = (_type != de::_DATA_TYPES_FLAGS_::_VOID_);

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->element_num = static_cast<uint64_t>(this->_layout.depth) * this->_layout.plane[0];
    this->_element_num = static_cast<uint64_t>(this->_layout.height) * this->_layout.dp_x_wp;
    this->total_bytes = this->_element_num * this->_layout._single_element_size;
}



void decx::_GPU_Tensor::alloc_data_space()
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        SetConsoleColor(4);
        printf("Internal error.\n");
        ResetConsoleColor;
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        SetConsoleColor(4);
        printf("Internal error.\n");
        ResetConsoleColor;
        return;
    }
    if (decx::alloc::_device_malloc(&this->Tens, this->total_bytes, true, S)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your device.");
        exit(-1);
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}




void decx::_GPU_Tensor::re_alloc_data_space(decx::cuda_stream* S)
{
    if (decx::alloc::_device_realloc(&this->Tens, this->total_bytes, true, S)) {
        Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your device.");
        exit(-1);
    }
}



void decx::_GPU_Tensor::construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}




void decx::_GPU_Tensor::re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth,
    decx::cuda_stream* S)
{
    if (this->type != _type || 
        this->_layout.width != _width || 
        this->_layout.height != _height || 
        this->_layout.depth != _depth) 
    {
        const uint64_t pre_size = this->total_bytes;
        
        this->_attribute_assign(_type, _width, _height, _depth);

        if (this->total_bytes > pre_size) {
            this->re_alloc_data_space(S);
        }
    }
}




decx::_GPU_Tensor::_GPU_Tensor(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    this->_attribute_assign(_type, _width, _height, _depth);

    this->alloc_data_space();
}




decx::_GPU_Tensor::_GPU_Tensor()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
    this->_init = false;
}




de::GPU_Tensor& decx::_GPU_Tensor::SoftCopy(de::GPU_Tensor& src)
{
    decx::_GPU_Tensor& ref_src = dynamic_cast<decx::_GPU_Tensor &>(src);

    this->Tens.block = ref_src.Tens.block;

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth);

    decx::alloc::_device_malloc_same_place(&this->Tens);

    return *this;
}



void decx::_GPU_Tensor::release()
{
    decx::alloc::_device_dealloc(&this->Tens);
}



_DECX_API_ de::GPU_Tensor& de::CreateGPUTensorRef()
{
    return *(new decx::_GPU_Tensor());
}



_DECX_API_ de::GPU_Tensor* de::CreateGPUTensorPtr()
{
    return new decx::_GPU_Tensor();
}



_DECX_API_ de::GPU_Tensor& de::CreateGPUTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    return *(new decx::_GPU_Tensor(_type, _width, _height, _depth));
}



_DECX_API_ de::GPU_Tensor* de::CreateGPUTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height, const uint32_t _depth)
{
    return new decx::_GPU_Tensor(_type, _width, _height, _depth);
}


de::_DATA_TYPES_FLAGS_ decx::_GPU_Tensor::Type() const
{
    return this->type;
}


void decx::_GPU_Tensor::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}


const decx::_tensor_layout& decx::_GPU_Tensor::get_layout() const
{
    return this->_layout;
}



bool decx::_GPU_Tensor::is_init() const
{
    return this->_init;
}


uint64_t decx::_GPU_Tensor::get_total_bytes() const
{
    return this->total_bytes;
}




_DECX_API_ de::DH de::cuda::PinMemory(de::Tensor& src)
{
    de::DH handle;

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    cudaError_t _err = cudaHostRegister(_src->Tens.ptr, _src->get_total_bytes(), cudaHostRegisterPortable);
    if (_err != cudaSuccess) {
        if (_err == cudaErrorHostMemoryAlreadyRegistered) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_HOST_MEM_REGISTERED, HOST_MEM_REGISTERED);
        }
        else {
            checkCudaErrors(_err);
        }
    }

    return handle;
}


_DECX_API_ de::DH de::cuda::UnpinMemory(de::Tensor& src)
{
    de::DH handle;

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    cudaError_t _err = cudaHostUnregister(_src->Tens.ptr);

    if (_err != cudaSuccess) {
        if (_err == cudaErrorHostMemoryNotRegistered) {
            decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_HOST_MEM_UNREGISTERED, HOST_MEM_UNREGISTERED);
        }
        else {
            checkCudaErrors(_err);
        }
    }

    return handle;
}