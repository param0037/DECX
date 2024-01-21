/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "MatrixArray.h"


void decx::_MatrixArray::alloc_data_space()
{
    if (decx::alloc::_host_virtual_page_malloc<void>(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
        return;
    }

    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->MatptrArr, this->ArrayNumber * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_single_element_size);
    }
}




void decx::_MatrixArray::re_alloc_data_space()
{
    if (decx::alloc::_host_virtual_page_realloc<void>(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
        return;
    }

    memset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->MatptrArr, this->ArrayNumber * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_single_element_size);
    }
}




void decx::_MatrixArray::construct(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint _MatrixNum)
{
    this->_attribute_assign(_type, _width, _height, _MatrixNum);

    this->alloc_data_space();
}



void decx::_MatrixArray::re_construct(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint _MatrixNum)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->_layout.width != _width || this->_layout.height != _height || 
        this->ArrayNumber != _MatrixNum)
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _MatrixNum);

        if (this->total_bytes > pre_size) {
            decx::alloc::_host_virtual_page_dealloc(&this->MatArr);

            this->alloc_data_space();
        }
        else {
            this->MatptrArr.ptr[0] = this->MatArr.ptr;
            for (int i = 1; i < this->ArrayNumber; ++i) {
                this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_single_element_size);
            }
        }
    }
}



void decx::_MatrixArray::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, uint _width, uint _height, uint MatrixNum)
{
    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);
    this->ArrayNumber = MatrixNum;

    this->_single_element_size = this->_layout._single_element_size;

    this->_init = true;

    this->_init = _type != de::_DATA_TYPES_FLAGS_::_VOID_;

    this->plane = static_cast<size_t>(_width) * static_cast<size_t>(_height);
    this->_plane = static_cast<size_t>(this->_layout.pitch) * static_cast<size_t>(this->_layout.height);

    this->element_num = static_cast<size_t>(this->plane) * static_cast<size_t>(MatrixNum);
    this->_element_num = static_cast<size_t>(this->_plane) * static_cast<size_t>(MatrixNum);

    this->total_bytes = (this->_element_num) * this->_layout._single_element_size;
}




decx::_MatrixArray::_MatrixArray()
{
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
    this->_init = false;
}



decx::_MatrixArray::_MatrixArray(const de::_DATA_TYPES_FLAGS_ _type, uint W, uint H, uint MatrixNum)
{
    this->_attribute_assign(_type, W, H, MatrixNum);

    this->alloc_data_space();
}


uint32_t decx::_MatrixArray::Width() const { return this->_layout.width; }


uint32_t decx::_MatrixArray::Height() const { return this->_layout.height; }


uint32_t decx::_MatrixArray::MatrixNumber() const { return this->ArrayNumber; }


float* decx::_MatrixArray::ptr_fp32(const uint row, const uint col, const uint _seq)
{
    float* __ptr = reinterpret_cast<float*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->_layout.pitch + col);
}

int* decx::_MatrixArray::ptr_int32(const uint row, const uint col, const uint _seq)
{
    int* __ptr = reinterpret_cast<int*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->_layout.pitch + col);
}

double* decx::_MatrixArray::ptr_fp64(const uint row, const uint col, const uint _seq)
{
    double* __ptr = reinterpret_cast<double*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->_layout.pitch + col);
}

de::CPf* decx::_MatrixArray::ptr_cpl32(const uint row, const uint col, const uint _seq)
{
    de::CPf* __ptr = reinterpret_cast<de::CPf*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->_layout.pitch + col);
}


de::Half* decx::_MatrixArray::ptr_fp16(const uint row, const uint col, const uint _seq)
{
    de::Half* __ptr = reinterpret_cast<de::Half*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->_layout.pitch + col);
}


uint8_t* decx::_MatrixArray::ptr_uint8(const uint row, const uint col, const uint _seq)
{
    uint8_t* __ptr = reinterpret_cast<uint8_t*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->_layout.pitch + col);
}





de::MatrixArray& de::CreateMatrixArrayRef()
{
    return *(new decx::_MatrixArray());
}



de::MatrixArray* de::CreateMatrixArrayPtr()
{
    return new decx::_MatrixArray();
}




de::MatrixArray& de::CreateMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum)
{
    return *(new decx::_MatrixArray(_type, width, height, MatrixNum));
}




de::MatrixArray* de::CreateMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum)
{
    return new decx::_MatrixArray(_type, width, height, MatrixNum);
}




void decx::_MatrixArray::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->MatArr);

    decx::alloc::_host_virtual_page_dealloc(&this->MatptrArr);
}



de::_DATA_TYPES_FLAGS_ decx::_MatrixArray::Type() const
{
    return this->type;
}


void decx::_MatrixArray::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}



de::MatrixArray& decx::_MatrixArray::SoftCopy(de::MatrixArray& src)
{
    const decx::_MatrixArray& ref_src = dynamic_cast<decx::_MatrixArray&>(src);

    this->MatArr.block = ref_src.MatArr.block;

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src.ArrayNumber);

    decx::alloc::_host_virtual_page_malloc_same_place(&this->MatArr);

    return *this;
}



uint32_t decx::_MatrixArray::Pitch() const
{
    return this->_layout.pitch;
}


uint32_t decx::_MatrixArray::Array_num() const
{
    return this->ArrayNumber;
}


const decx::_matrix_layout& decx::_MatrixArray::get_layout() const
{
    return this->_layout;
}




bool decx::_MatrixArray::is_init() const
{
    return this->_init;
}


uint64_t decx::_MatrixArray::get_total_bytes() const
{
    return this->total_bytes;
}