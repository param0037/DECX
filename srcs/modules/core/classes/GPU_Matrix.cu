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


#include "../../../common/Classes/GPU_Matrix.h"


void decx::_matrix_layout::_attribute_assign(const de::_DATA_TYPES_FLAGS_ type, const uint _width, const uint _height)
{
    this->_single_element_size = decx::core::_size_mapping(type);

    this->width = _width;
    this->height = _height;

    uint32_t _alignment = 1;
    if (type != de::_DATA_TYPES_FLAGS_::_VOID_)
    {
        // To make memory access coalesced on device memory, the alignment should be 128 byte
        switch (this->_single_element_size)
        {
        case 4:
            _alignment = 32;     break;
        case 8:
            _alignment = 16;     break;
        case 2:
            _alignment = 64;     break;
        case 1:
            _alignment = 128;     break;
        default:
            break;
        }
    }
    this->pitch = decx::utils::ceil<uint>(_width, _alignment) * _alignment;
}



void decx::_GPU_Matrix::alloc_data_space()
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
    
    if (decx::alloc::_device_malloc(&this->Mat, this->total_bytes, true, S)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.\n");
        ResetConsoleColor;
        return;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



void decx::_GPU_Matrix::re_alloc_data_space(decx::cuda_stream* S)
{
    if (decx::alloc::_device_realloc(&this->Mat, this->total_bytes, true, S)) {
        return;
    }
}



void decx::_GPU_Matrix::construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t _width, uint32_t _height,
    const de::_DATA_FORMATS_ format)
{
    this->_attribute_assign(_type, _width, _height);
    this->_format = format;

    this->alloc_data_space();
}



void decx::_GPU_Matrix::re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint32_t _width, uint32_t _height,
    decx::cuda_stream* S)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->_layout.width != _width || this->_layout.height != _height)
    {
        const size_t pre_size = this->total_bytes;
        this->_attribute_assign(_type, _width, _height);

        if (this->total_bytes > pre_size) {
            this->re_alloc_data_space(S);
        }
    }

}


void decx::_GPU_Matrix::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height)
{
    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);

    this->_layout.pitch = this->_layout.pitch;
    this->_init = (_type != de::_DATA_TYPES_FLAGS_::_VOID_);

    uint64_t element_num = static_cast<size_t>(this->_layout.pitch) * static_cast<uint64_t>(_height);

    this->total_bytes = (element_num)*this->_layout._single_element_size;
}



uint32_t decx::_GPU_Matrix::Width() const
{
    return this->_layout.width;
}


uint32_t decx::_GPU_Matrix::Height() const
{
    return this->_layout.height;
}


decx::_GPU_Matrix::_GPU_Matrix(const de::_DATA_TYPES_FLAGS_ _type, const uint32_t _width, const uint32_t _height,
    const de::_DATA_FORMATS_ format)
{
    this->_attribute_assign(_type, _width, _height);
    this->_format = format;

    this->alloc_data_space();
}



decx::_GPU_Matrix::_GPU_Matrix()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0);
    this->_init = false;
}



void decx::_GPU_Matrix::release()
{
    decx::alloc::_device_dealloc(&this->Mat);
}



de::_DATA_TYPES_FLAGS_ decx::_GPU_Matrix::Type() const
{
    return this->type;
}


void decx::_GPU_Matrix::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}



de::GPU_Matrix& decx::_GPU_Matrix::SoftCopy(de::GPU_Matrix& src)
{
    decx::_GPU_Matrix& ref_src = dynamic_cast<decx::_GPU_Matrix&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height);
    decx::alloc::_device_malloc_same_place(&this->Mat);

    return *this;
}


de::_DATA_FORMATS_ decx::_GPU_Matrix::Format() const
{
    return this->_format;
}



uint32_t decx::_GPU_Matrix::Pitch() const
{
    return this->_layout.pitch;
}


const decx::_matrix_layout& decx::_GPU_Matrix::get_layout() const
{
    return this->_layout;
}


bool decx::_GPU_Matrix::is_init() const
{
    return this->_init;
}


uint64_t decx::_GPU_Matrix::get_total_bytes() const
{
    return this->total_bytes;
}



de::_DATA_FORMATS_ decx::_GPU_Matrix::get_data_format() const
{
    return this->_format;
}


void decx::_GPU_Matrix::set_data_format(const de::_DATA_FORMATS_& format)
{
    this->_format = format;
}


de::GPU_Matrix& de::CreateGPUMatrixRef()
{
    return *(new decx::_GPU_Matrix());
}


de::GPU_Matrix* de::CreateGPUMatrixPtr()
{
    return new decx::_GPU_Matrix();
}



de::GPU_Matrix& de::CreateGPUMatrixRef(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height,
    const de::_DATA_FORMATS_ format)
{
    return *(new decx::_GPU_Matrix(_type, width, height, format));
}



de::GPU_Matrix* de::CreateGPUMatrixPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height,
    const de::_DATA_FORMATS_ format)
{
    return new decx::_GPU_Matrix(_type, width, height, format);
}



_DECX_API_ de::DH de::cuda::PinMemory(de::Matrix& src)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    cudaError_t _err = cudaHostRegister(_src->Mat.ptr, _src->get_total_bytes(), cudaHostRegisterPortable);
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


_DECX_API_ de::DH de::cuda::UnpinMemory(de::Matrix& src)
{
    de::DH handle;

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    cudaError_t _err = cudaHostUnregister(_src->Mat.ptr);

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