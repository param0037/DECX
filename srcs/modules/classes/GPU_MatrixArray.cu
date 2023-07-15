/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "GPU_MatrixArray.h"



uint32_t decx::_GPU_MatrixArray::Width() { return this->_layout.width; }


uint32_t decx::_GPU_MatrixArray::Height() { return this->_layout.height; }


uint32_t decx::_GPU_MatrixArray::MatrixNumber() { return this->ArrayNumber; }



void decx::_GPU_MatrixArray::re_alloc_data_space()
{
    if (decx::alloc::_device_realloc(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    cudaMemset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_realloc<void*>(&this->MatptrArr, this->ArrayNumber * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_layout._single_element_size);
    }
}



void decx::_GPU_MatrixArray::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    cudaMemset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->MatptrArr, this->ArrayNumber * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_layout._single_element_size);
    }
}




void decx::_GPU_MatrixArray::_attribute_assign(const int _type, uint _width, uint _height, uint MatrixNum)
{
    /*this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

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

    this->pitch = decx::utils::ceil<size_t>(_width, _alignment) * _alignment;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->pitch) * static_cast<size_t>(this->height);

    this->element_num = this->plane * MatrixNum;
    this->_element_num = this->_plane * MatrixNum;

    this->total_bytes = this->_element_num * this->_single_element_size;*/

    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);
    this->ArrayNumber = MatrixNum;

    //this->_layout._single_element_size = this->_layout._single_element_size;

    this->_init = true;

    //this->width = this->_layout.width;
    //this->height = this->_layout.height;

    if (_type != decx::_DATA_TYPES_FLAGS_::_VOID_)
    {
        //this->pitch = this->_layout.pitch;
        this->_init = true;

        this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
        this->_plane = static_cast<size_t>(this->_layout.pitch) * static_cast<size_t>(this->_layout.height);

        this->element_num = static_cast<size_t>(this->plane) * static_cast<size_t>(MatrixNum); 
        this->_element_num = static_cast<size_t>(this->_plane) * static_cast<size_t>(MatrixNum);

        this->total_bytes = (this->_element_num) * this->_layout._single_element_size;
    }
}



void decx::_GPU_MatrixArray::construct(const int _type, uint _width, uint _height, uint MatrixNum)
{
    this->_attribute_assign(_type, _width, _height, MatrixNum);

    this->alloc_data_space();
}



void decx::_GPU_MatrixArray::re_construct(const int _type, uint _width, uint _height, uint MatrixNum)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->_layout.width != _width || this->_layout.height != _height || 
        this->ArrayNumber != MatrixNum)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, MatrixNum);

        if (this->total_bytes > pre_size)
        {
            decx::alloc::_host_virtual_page_dealloc(&this->MatptrArr);
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
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
}



decx::_GPU_MatrixArray::_GPU_MatrixArray(const int _type, uint W, uint H, uint MatrixNum)
{
    this->_attribute_assign(_type, W, H, MatrixNum);

    this->alloc_data_space();
}





void decx::_GPU_MatrixArray::release()
{
    decx::alloc::_device_dealloc(&this->MatArr);
}




namespace de
{
    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef();


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr();


    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef(const int _type, const uint _width, const uint _height, const uint _Mat_number);


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr(const int _type, const uint _width, const uint _height, const uint _Mat_number);
}




de::GPU_MatrixArray& de::CreateGPUMatrixArrayRef()
{
    return *(new decx::_GPU_MatrixArray());
}




de::GPU_MatrixArray* de::CreateGPUMatrixArrayPtr()
{
    return new decx::_GPU_MatrixArray();
}




de::GPU_MatrixArray& de::CreateGPUMatrixArrayRef(const int _type, const uint _width, const uint _height, const uint _Mat_number)
{
    return *(new decx::_GPU_MatrixArray(_type, _width, _height, _Mat_number));
}




de::GPU_MatrixArray* de::CreateGPUMatrixArrayPtr(const int _type, const uint _width, const uint _height, const uint _Mat_number)
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


int decx::_GPU_MatrixArray::Type()
{
    return this->type;
}



uint32_t decx::_GPU_MatrixArray::Pitch()
{
    return this->_layout.pitch;
}


const decx::_matrix_layout& decx::_GPU_MatrixArray::get_layout()
{
    return this->_layout;
}


bool decx::_GPU_MatrixArray::is_init()
{
    return this->_init;
}