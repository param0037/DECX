/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "GPU_Matrix.h"


void decx::_GPU_Matrix::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->Mat, this->_total_bytes)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }

    cudaMemset(this->Mat.ptr, 0, this->total_bytes);
}



void decx::_GPU_Matrix::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->Mat, this->_total_bytes)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.");
        ResetConsoleColor;
        return;
    }

    cudaMemset(this->Mat.ptr, 0, this->total_bytes);
}



void decx::_GPU_Matrix::construct(const int _type, uint _width, uint _height)
{
    this->_attribute_assign(_type, _width, _height);

    this->alloc_data_space();
}



void decx::_GPU_Matrix::re_construct(const int _type, uint _width, uint _height)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->width != _width || this->height != _height)
    {
        this->_attribute_assign(_type, _width, _height);

        this->re_alloc_data_space();
    }
}


void decx::_GPU_Matrix::_attribute_assign(const int _type, const uint _width, const uint _height)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    uint _alignment = 0;
    switch (this->_single_element_size)
    {
    case 4:
        _alignment = _MATRIX_ALIGN_4B_;     break;
    case 8:
        _alignment = _MATRIX_ALIGN_8B_;     break;
    case 2:
        _alignment = _MATRIX_ALIGN_2B_;     break;
    case 1:
        _alignment = _MATRIX_ALIGN_1B_;     break;
    default:
        break;
    }
    this->pitch = decx::utils::ceil<int>(_width, _alignment) * _alignment;

    this->element_num = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->total_bytes = (this->element_num) * this->_single_element_size;

    this->_element_num = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);
    this->_total_bytes = (this->_element_num) * this->_single_element_size;
}




decx::_GPU_Matrix::_GPU_Matrix(const int _type, const uint _width, const uint _height)
{
    this->_attribute_assign(_type, _width, _height);

    this->alloc_data_space();
}



decx::_GPU_Matrix::_GPU_Matrix()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0);
}



void decx::_GPU_Matrix::release()
{
    decx::alloc::_device_dealloc(&this->Mat);
}



int decx::_GPU_Matrix::Type()
{
    return this->type;
}



void decx::_GPU_Matrix::Load_from_host(de::Matrix& src)
{
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    checkCudaErrors(cudaMemcpy2D(this->Mat.ptr, this->pitch * this->_single_element_size,
        _src->Mat.ptr, _src->pitch * _src->_single_element_size, this->width * this->_single_element_size, this->height,
        cudaMemcpyHostToDevice));
}



void decx::_GPU_Matrix::Load_to_host(de::Matrix& dst)
{
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    checkCudaErrors(cudaMemcpy2D(_dst->Mat.ptr, _dst->pitch * this->_single_element_size,
        this->Mat.ptr, this->pitch * this->_single_element_size, _dst->pitch * _dst->_single_element_size, _dst->height,
        cudaMemcpyDeviceToHost));
}


de::GPU_Matrix& decx::_GPU_Matrix::SoftCopy(de::GPU_Matrix& src)
{
    decx::_GPU_Matrix& ref_src = dynamic_cast<decx::_GPU_Matrix&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.type, ref_src.width, ref_src.height);
    decx::alloc::_device_malloc_same_place(&this->Mat);

    return *this;
}


namespace de
{
    _DECX_API_ de::GPU_Matrix& CreateGPUMatrixRef();


    _DECX_API_ de::GPU_Matrix* CreateGPUMatrixPtr();


    _DECX_API_ de::GPU_Matrix& CreateGPUMatrixRef(const int _type, const uint width, const uint height);


    _DECX_API_ de::GPU_Matrix* CreateGPUMatrixPtr(const int _type, const uint width, const uint height);
}



de::GPU_Matrix& de::CreateGPUMatrixRef()
{
    return *(new decx::_GPU_Matrix());
}


de::GPU_Matrix* de::CreateGPUMatrixPtr()
{
    return new decx::_GPU_Matrix();
}



de::GPU_Matrix& de::CreateGPUMatrixRef(const int _type, const uint width, const uint height)
{
    return *(new decx::_GPU_Matrix(_type, width, height));
}




de::GPU_Matrix* de::CreateGPUMatrixPtr(const int _type, const uint width, const uint height)
{
    return new decx::_GPU_Matrix(_type, width, height);
}