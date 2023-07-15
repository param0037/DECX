/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "GPU_Matrix.h"



void decx::_matrix_layout::_attribute_assign(const int type, const uint _width, const uint _height)
{
    this->_single_element_size = decx::core::_size_mapping(type);

    this->width = _width;
    this->height = _height;

    if (type != decx::_DATA_TYPES_FLAGS_::_VOID_)
    {
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
        this->pitch = decx::utils::ceil<uint>(_width, _alignment) * _alignment;
    }
}



void decx::_GPU_Matrix::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->Mat, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.\n");
        ResetConsoleColor;
        return;
    }

    cudaMemset(this->Mat.ptr, 0, this->total_bytes);
}



void decx::_GPU_Matrix::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->Mat, this->total_bytes)) {
        SetConsoleColor(4);
        printf("Matrix malloc failed! Please check if there is enough space in your device.\n");
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
    if (this->_layout.width != _width || this->_layout.height != _height)
    {
        const size_t pre_size = this->total_bytes;
        this->_attribute_assign(_type, _width, _height);

        if (this->total_bytes > pre_size) {
            this->re_alloc_data_space();
        }
    }
}


void decx::_GPU_Matrix::_attribute_assign(const int _type, const uint _width, const uint _height)
{
    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);

    if (_type != decx::_DATA_TYPES_FLAGS_::_VOID_)
    {
        this->_layout.pitch = this->_layout.pitch;
        this->_init = true;

        uint64_t element_num = static_cast<size_t>(this->_layout.pitch) * static_cast<size_t>(_height);

        this->total_bytes = (element_num) * this->_layout._single_element_size;
    }
}



uint32_t decx::_GPU_Matrix::Width()
{
    return this->_layout.width;
}


uint32_t decx::_GPU_Matrix::Height()
{
    return this->_layout.height;
}


decx::_GPU_Matrix::_GPU_Matrix(const int _type, const uint _width, const uint _height)
{
    this->_attribute_assign(_type, _width, _height);

    this->alloc_data_space();
}



decx::_GPU_Matrix::_GPU_Matrix()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0);
    this->_init = false;
}



void decx::_GPU_Matrix::release()
{
    decx::alloc::_device_dealloc(&this->Mat);
}



int decx::_GPU_Matrix::Type()
{
    return this->type;
}




de::GPU_Matrix& decx::_GPU_Matrix::SoftCopy(de::GPU_Matrix& src)
{
    decx::_GPU_Matrix& ref_src = dynamic_cast<decx::_GPU_Matrix&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height);
    decx::alloc::_device_malloc_same_place(&this->Mat);

    return *this;
}


uint32_t decx::_GPU_Matrix::Pitch()
{
    return this->_layout.pitch;
}


const decx::_matrix_layout& decx::_GPU_Matrix::get_layout()
{
    return this->_layout;
}


bool decx::_GPU_Matrix::is_init()
{
    return this->_init;
}


uint64_t decx::_GPU_Matrix::get_total_bytes()
{
    return this->total_bytes;
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