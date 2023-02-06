/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "GPU_Vector.h"



void decx::_GPU_Vector::_attribute_assign(const int _type, size_t length)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    uint _alignment = 0;
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
    this->_init = true;
    this->_length = decx::utils::ceil<size_t>(length, (size_t)_alignment) * (size_t)_alignment;
    this->total_bytes = this->_length * this->_single_element_size;
}



int decx::_GPU_Vector::Type()
{
    return this->type;
}


void decx::_GPU_Vector::alloc_data_space()
{
    if (decx::alloc::_device_malloc<void>(&this->Vec, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Vector on GPU malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }
}


void decx::_GPU_Vector::construct(const int _type, size_t length)
{
    this->_attribute_assign(_type, length);

    this->alloc_data_space();
}


void decx::_GPU_Vector::re_construct(const int _type, size_t length)
{
    if (this->type != _type || this->length != _length) {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, length);

        if (this->total_bytes > pre_size) {
            this->alloc_data_space();
        }
    }
}


decx::_GPU_Vector::_GPU_Vector(const int _type, size_t length)
{
    this->construct(_type, length);
}



decx::_GPU_Vector::_GPU_Vector()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0);
    this->_init = false;
}



void decx::_GPU_Vector::load_from_host(de::Vector& src)
{
    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);
    checkCudaErrors(cudaMemcpy(this->Vec.ptr, _src->Vec.ptr, this->length * this->_single_element_size, cudaMemcpyHostToDevice));
}



void decx::_GPU_Vector::load_to_host(de::Vector& dst)
{
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);
    checkCudaErrors(cudaMemcpy(_dst->Vec.ptr, this->Vec.ptr, this->length * this->_single_element_size, cudaMemcpyDeviceToHost));
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



namespace de
{
    _DECX_API_ de::GPU_Vector& CreateGPUVectorRef();


    _DECX_API_ de::GPU_Vector* CreateGPUVectorPtr();


    _DECX_API_ de::GPU_Vector& CreateGPUVectorRef(const int _type, const size_t length);


    _DECX_API_ de::GPU_Vector* CreateGPUVectorPtr(const int _type, const size_t length);
}



de::GPU_Vector& de::CreateGPUVectorRef() {
    return *(new decx::_GPU_Vector());
}



de::GPU_Vector* de::CreateGPUVectorPtr() {
    return new decx::_GPU_Vector();
}




de::GPU_Vector& de::CreateGPUVectorRef(const int _type, const size_t length) {
    return *(new decx::_GPU_Vector(_type, length));
}



de::GPU_Vector* de::CreateGPUVectorPtr(const int _type, const size_t length) {
    return new decx::_GPU_Vector(_type, length);
}
