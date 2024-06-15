/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "GPU_Vector.h"



void decx::_GPU_Vector::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    uint32_t _alignment = 1;
    switch (this->_single_element_size)
    {
    case _SIZE_INT32_:
        _alignment = _VECTOR_ALIGN_4B_;     break;
    case _SIZE_FLOAT64_:
        _alignment = _VECTOR_ALIGN_8B_;     break;
    case _SIZE_FLOAT16_:
        _alignment = _VECTOR_ALIGN_2B_;     break;
    case _SIZE_UINT8_:
        _alignment = _VECTOR_ALIGN_1B_;     break;
    default:
        break;
    }
    this->length = length;
    this->_init = (_type != de::_DATA_TYPES_FLAGS_::_VOID_);
    this->_length = decx::utils::ceil<size_t>(length, (size_t)_alignment) * (size_t)_alignment;
    this->total_bytes = this->_length * this->_single_element_size;
}



de::_DATA_TYPES_FLAGS_ decx::_GPU_Vector::Type() const
{
    return this->type;
}


bool decx::_GPU_Vector::is_init() const
{
    return this->_init;
}


uint64_t decx::_GPU_Vector::_Length() const
{
    return this->_length;
}

uint64_t decx::_GPU_Vector::get_total_bytes() const
{
    return this->total_bytes;
}


void decx::_GPU_Vector::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}


decx::_GPU_Vector::~_GPU_Vector()
{
    this->release();
}


void decx::_GPU_Vector::alloc_data_space()
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    if (decx::alloc::_device_malloc<void>(&this->Vec, this->total_bytes, true, S)) {
        SetConsoleColor(4);
        printf("Vector on GPU malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}


void decx::_GPU_Vector::re_alloc_data_space()
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    if (decx::alloc::_device_realloc<void>(&this->Vec, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Vector on GPU malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }

    checkCudaErrors(cudaMemsetAsync(this->Vec.ptr, 0, this->total_bytes, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}


void decx::_GPU_Vector::construct(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    this->_attribute_assign(_type, length);

    this->alloc_data_space();
}


void decx::_GPU_Vector::re_construct(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    if (this->type != _type || this->length != _length) {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, length);

        if (this->total_bytes > pre_size) {
            this->alloc_data_space();
        }
    }
}


decx::_GPU_Vector::_GPU_Vector(const de::_DATA_TYPES_FLAGS_ _type, size_t length)
{
    this->construct(_type, length);
}



decx::_GPU_Vector::_GPU_Vector()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0);
    this->_init = false;
}


uint64_t decx::_GPU_Vector::Len() const
{
    return this->length;
}


void decx::_GPU_Vector::release()
{
    decx::alloc::_device_dealloc(&this->Vec);
}



de::GPU_Vector& decx::_GPU_Vector::SoftCopy(de::GPU_Vector& src)
{
    const decx::_GPU_Vector& ref_src = dynamic_cast<decx::_GPU_Vector&>(src);

    this->_attribute_assign(ref_src.type, ref_src.length);
    decx::alloc::_device_malloc_same_place(&this->Vec);

    return *this;
}



de::GPU_Vector& de::CreateGPUVectorRef() {
    return *(new decx::_GPU_Vector());
}



de::GPU_Vector* de::CreateGPUVectorPtr() {
    return new decx::_GPU_Vector();
}



de::GPU_Vector& de::CreateGPUVectorRef(const de::_DATA_TYPES_FLAGS_ _type, const size_t length) {
    return *(new decx::_GPU_Vector(_type, length));
}



de::GPU_Vector* de::CreateGPUVectorPtr(const de::_DATA_TYPES_FLAGS_ _type, const size_t length) {
    return new decx::_GPU_Vector(_type, length);
}




_DECX_API_ de::DH de::cuda::PinMemory(de::Vector& src)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    cudaError_t _err = cudaHostRegister(_src->Vec.ptr, _src->get_total_bytes(), cudaHostRegisterPortable);
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


_DECX_API_ de::DH de::cuda::UnpinMemory(de::Vector& src)
{
    de::DH handle;

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    cudaError_t _err = cudaHostUnregister(_src->Vec.ptr);

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