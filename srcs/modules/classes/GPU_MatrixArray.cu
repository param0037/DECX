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

#include "GPU_MatrixArray.h"



uint32_t decx::_GPU_MatrixArray::Width() const { return this->_layout.width; }


uint32_t decx::_GPU_MatrixArray::Height() const { return this->_layout.height; }


uint32_t decx::_GPU_MatrixArray::MatrixNumber() const { return this->ArrayNumber; }



void decx::_GPU_MatrixArray::re_alloc_data_space()
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

    if (decx::alloc::_device_realloc(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    checkCudaErrors(cudaMemsetAsync(this->MatArr.ptr, 0, this->total_bytes, S->get_raw_stream_ref()));

    this->MatptrArr.ptr = (void**)realloc(this->MatptrArr.ptr, this->ArrayNumber * sizeof(void*));
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_layout._single_element_size);
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



void decx::_GPU_MatrixArray::alloc_data_space()
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

    if (decx::alloc::_device_malloc(&this->MatArr, this->total_bytes, true, S)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    this->MatptrArr.ptr = (void**)malloc(this->ArrayNumber * sizeof(void*));
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_layout._single_element_size);
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}


void decx::_GPU_MatrixArray::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint MatrixNum)
{
    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);
    this->ArrayNumber = MatrixNum;

    this->_init = _type != de::_DATA_TYPES_FLAGS_::_VOID_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->_layout.pitch) * static_cast<size_t>(this->_layout.height);

    this->element_num = static_cast<size_t>(this->plane) * static_cast<size_t>(MatrixNum);
    this->_element_num = static_cast<size_t>(this->_plane) * static_cast<size_t>(MatrixNum);

    this->total_bytes = (this->_element_num) * this->_layout._single_element_size;
}



void decx::_GPU_MatrixArray::construct(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint MatrixNum)
{
    this->_attribute_assign(_type, _width, _height, MatrixNum);

    this->alloc_data_space();
}



void decx::_GPU_MatrixArray::re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint MatrixNum)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->_layout.width != _width || this->_layout.height != _height || 
        this->ArrayNumber != MatrixNum)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, MatrixNum);

        if (this->total_bytes > pre_size)
        {
            free(this->MatptrArr.ptr);
            if (this->MatArr.ptr == NULL) {
                decx::alloc::_device_dealloc(&this->MatArr);
                this->alloc_data_space();
            }
            else {
                this->re_alloc_data_space();
            }
        }
        else {
            this->MatptrArr.ptr[0] = this->MatArr.ptr;
            for (int i = 1; i < this->ArrayNumber; ++i) {
                this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_layout._single_element_size);
            }
        }
    }
}



decx::_GPU_MatrixArray::_GPU_MatrixArray()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
}



decx::_GPU_MatrixArray::_GPU_MatrixArray(const de::_DATA_TYPES_FLAGS_ _type, uint W, uint H, uint MatrixNum)
{
    this->_attribute_assign(_type, W, H, MatrixNum);

    this->alloc_data_space();
}



void decx::_GPU_MatrixArray::release()
{
    decx::alloc::_device_dealloc(&this->MatArr);
    free(this->MatptrArr.ptr);
}




de::GPU_MatrixArray& de::CreateGPUMatrixArrayRef()
{
    return *(new decx::_GPU_MatrixArray());
}




de::GPU_MatrixArray* de::CreateGPUMatrixArrayPtr()
{
    return new decx::_GPU_MatrixArray();
}




de::GPU_MatrixArray& de::CreateGPUMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _Mat_number)
{
    return *(new decx::_GPU_MatrixArray(_type, _width, _height, _Mat_number));
}




de::GPU_MatrixArray* de::CreateGPUMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _Mat_number)
{
    return new decx::_GPU_MatrixArray(_type, _width, _height, _Mat_number);
}



de::GPU_MatrixArray& decx::_GPU_MatrixArray::SoftCopy(de::GPU_MatrixArray& src)
{
    const decx::_GPU_MatrixArray& ref_src = dynamic_cast<decx::_GPU_MatrixArray&>(src);

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src.ArrayNumber);
    decx::alloc::_device_malloc_same_place(&this->MatArr);

    return *this;
}


de::_DATA_TYPES_FLAGS_ decx::_GPU_MatrixArray::Type() const
{
    return this->type;
}


void decx::_GPU_MatrixArray::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}



uint32_t decx::_GPU_MatrixArray::Pitch() const
{
    return this->_layout.pitch;
}


const decx::_matrix_layout& decx::_GPU_MatrixArray::get_layout() const
{
    return this->_layout;
}


bool decx::_GPU_MatrixArray::is_init() const
{
    return this->_init;
}


uint64_t decx::_GPU_MatrixArray::get_total_bytes() const
{
    return this->total_bytes;
}



_DECX_API_ de::DH de::cuda::PinMemory(de::MatrixArray& src)
{
    de::DH handle;

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    cudaError_t _err = cudaHostRegister(_src->MatArr.ptr, _src->get_total_bytes(), cudaHostRegisterPortable);
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


_DECX_API_ de::DH de::cuda::UnpinMemory(de::MatrixArray& src)
{
    de::DH handle;

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    cudaError_t _err = cudaHostUnregister(_src->MatArr.ptr);

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