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

#include <Classes/GPU_TensorArray.h>

#define MODULE_TAG "GPU_TensorArray"

void decx::_GPU_TensorArray::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->type = _type;

    this->type = _type;
    this->tensor_num = _tensor_num;
    this->_init = (_type != de::_DATA_TYPES_FLAGS_::_VOID_);

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->_gap = this->_layout.dp_x_wp * static_cast<uint64_t>(this->_layout.height);

    this->element_num = static_cast<uint64_t>(this->_layout.depth) * this->_layout.plane[0] * static_cast<uint64_t>(this->tensor_num);
    this->_element_num = static_cast<uint64_t>(this->tensor_num) * this->_gap;
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


de::_DATA_TYPES_FLAGS_ decx::_GPU_TensorArray::Type() const
{
    return this->type;
}


void decx::_GPU_TensorArray::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}



const decx::_tensor_layout& decx::_GPU_TensorArray::get_layout() const
{
    return this->_layout;
}


void decx::_GPU_TensorArray::alloc_data_space()
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        DECX_LOG_ERR("Failed to get cuda stream from queue");
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        DECX_LOG_ERR("Failed to get cuda event from queue");
        return;
    }

    if (this->TensArr.Allocate(this->total_bytes, CUDA_DEVICE, de::GetLastError(), true, S)) {
        DECX_LOG_ERR("Fail to allocate memory for GPU_TensorArray on device");
        exit(-1);
    }

    this->TensptrArr.Allocate(this->tensor_num * sizeof(void*), PAGABLE);
    this->TensptrArr[0] = this->TensArr.GetRawPtr();

    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr[i] = (uint8_t*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



void decx::_GPU_TensorArray::re_alloc_data_space()
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        DECX_LOG_ERR("Failed to get cuda stream from queue");
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        DECX_LOG_ERR("Failed to get cuda event from queue");
        return;
    }

    if (this->TensArr.Allocate(this->total_bytes, CUDA_DEVICE, de::GetLastError(), true, S)) {
        DECX_LOG_ERR("Fail to re-allocate memory for GPU_TensorArray on device\n");
        return;
    }

    this->TensptrArr.Allocate(this->tensor_num * sizeof(void*), PAGABLE);

    this->TensptrArr[0] = this->TensArr.GetRawPtr();
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr[i] = (uint8_t*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



void decx::_GPU_TensorArray::construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}



void decx::_GPU_TensorArray::re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    if (this->type != _type || this->_layout.width != _width || this->_layout.height != _height || 
        this->_layout.depth != _depth || this->tensor_num != _tensor_num)
    {
        const uint64_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

        if (this->total_bytes > pre_size)
        {
            this->TensptrArr.Free();

            if (this->TensArr.IsValid()) {
                this->TensArr.Free();

                this->alloc_data_space();
            }
            else {
                this->re_alloc_data_space();
            }
        }
        else {
            this->TensptrArr[0] = this->TensArr.GetRawPtr();
            for (uint i = 1; i < this->tensor_num; ++i) {
                this->TensptrArr[i] = (void*)((uchar*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size);
            }
        }
    }
}


decx::_GPU_TensorArray::_GPU_TensorArray()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0);

    this->_init = false;
}


decx::_GPU_TensorArray::_GPU_TensorArray(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}


de::GPU_TensorArray& decx::_GPU_TensorArray::SoftCopy(const de::GPU_TensorArray& src)
{
    const decx::_GPU_TensorArray& ref_src = dynamic_cast<const decx::_GPU_TensorArray&>(src);

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth, ref_src.tensor_num);

    this->TensArr.AllocateRef();

    this->TensptrArr.Reallocate(this->tensor_num * sizeof(void*), de::GetLastError());
    this->TensptrArr[0] = this->TensArr.GetRawPtr();
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr[i] = (uint8_t*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size;
    }

    return *this;
}



de::DH decx::_GPU_TensorArray::Extract_SoftCopy(const uint32_t index, de::GPU_Tensor& dst) const
{
    de::DH handle;
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);

    if (index > this->TensorNum() - 1) {
        DecxAssignLastHandle(&handle, DecxErrorTypes_e::DECX_FAIL_DimsNotMatching,
            "Overrange\n");
        return handle;
    }

    _dst->_attribute_assign(this->Type(), this->Width(), this->Height(), this->Depth());
    // _dst->Tens.ptr = this->TensptrArr.ptr[index];

    return handle;
}



void decx::_GPU_TensorArray::release()
{
    this->TensArr.Free();
    this->TensptrArr.Free();
}


bool decx::_GPU_TensorArray::is_init() const
{
    return this->_init;
}


uint64_t decx::_GPU_TensorArray::get_total_bytes() const
{
    return this->total_bytes;
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
de::GPU_TensorArray& de::CreateGPUTensorArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num)
{
    return *(new decx::_GPU_TensorArray(_type, width, height, depth, tensor_num));
}


_DECX_API_
de::GPU_TensorArray* de::CreateGPUTensorArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num)
{
    return new decx::_GPU_TensorArray(_type, width, height, depth, tensor_num);
}



_DECX_API_ de::DH de::cuda::PinMemory(de::TensorArray& src)
{
    de::DH handle;

    decx::_TensorArray* _src = dynamic_cast<decx::_TensorArray*>(&src);
    cudaError_t _err = cudaHostRegister(_src->TensArr.GetRawPtr(), _src->get_total_bytes(), cudaHostRegisterPortable);
    if (_err != cudaSuccess) {
        if (_err == cudaErrorHostMemoryAlreadyRegistered) {
            DecxAssignLastHandle(&handle, DecxErrorTypes_e::DECX_FAIL_HOST_MEM_REGISTERED, HOST_MEM_REGISTERED);
        }
        else {
            checkCudaErrors(_err);
        }
    }

    return handle;
}


_DECX_API_ de::DH de::cuda::UnpinMemory(de::TensorArray& src)
{
    de::DH handle;

    decx::_TensorArray* _src = dynamic_cast<decx::_TensorArray*>(&src);
    cudaError_t _err = cudaHostUnregister(_src->TensArr.GetRawPtr());

    if (_err != cudaSuccess) {
        if (_err == cudaErrorHostMemoryNotRegistered) {
            DecxAssignLastHandle(&handle, DecxErrorTypes_e::DECX_FAIL_HOST_MEM_UNREGISTERED, HOST_MEM_UNREGISTERED);
        }
        else {
            checkCudaErrors(_err);
        }
    }

    return handle;
}
