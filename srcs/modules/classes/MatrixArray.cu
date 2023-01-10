/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "MatrixArray.h"


void decx::_MatrixArray::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<void>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<void>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;

    default:
        break;
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




void decx::_MatrixArray::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<void>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<void>(&this->MatArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for MatrixArray on host\n");
            return;
        }
        break;

    default:
        break;
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




void decx::_MatrixArray::construct(const int _type, uint _width, uint _height, uint _MatrixNum, const int _flag)
{
    this->_attribute_assign(_type, _width, _height, _MatrixNum, _flag);

    this->alloc_data_space();
}



void decx::_MatrixArray::re_construct(const int _type, uint _width, uint _height, uint _MatrixNum, const int _flag)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->width != _width || this->height != _height || 
        this->ArrayNumber != _MatrixNum || this->_store_type != _flag)
    {
        decx::alloc::_host_virtual_page_dealloc(&this->MatptrArr);
        if (this->_store_type != _flag) {
            switch (this->_store_type)
            {
            case decx::DATA_STORE_TYPE::Page_Default:
                decx::alloc::_host_virtual_page_dealloc(&this->MatArr);
                break;

            case decx::DATA_STORE_TYPE::Page_Locked:
                decx::alloc::_host_fixed_page_dealloc(&this->MatArr);
                break;
            default:
                break;
            }
            this->_attribute_assign(_type, _width, _height, _MatrixNum, _flag);
            this->alloc_data_space();
        }
        else {
            this->_attribute_assign(_type, _width, _height, _MatrixNum, _flag);
            this->re_alloc_data_space();
        }
    }
}



void decx::_MatrixArray::_attribute_assign(const int _type, uint _width, uint _height, uint MatrixNum, const int flag)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->ArrayNumber = MatrixNum;

    this->_store_type = flag;

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

    this->total_bytes = this->_element_num * sizeof(float);
}




decx::_MatrixArray::_MatrixArray()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0);
}



decx::_MatrixArray::_MatrixArray(const int _type, uint W, uint H, uint MatrixNum, const int flag)
{
    this->_attribute_assign(_type, W, H, MatrixNum, flag);

    this->alloc_data_space();
}



float* decx::_MatrixArray::ptr_fp32(const uint row, const uint col, const uint _seq)
{
    float* __ptr = reinterpret_cast<float*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->pitch + col);
}

int* decx::_MatrixArray::ptr_int32(const uint row, const uint col, const uint _seq)
{
    int* __ptr = reinterpret_cast<int*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->pitch + col);
}

double* decx::_MatrixArray::ptr_fp64(const uint row, const uint col, const uint _seq)
{
    double* __ptr = reinterpret_cast<double*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->pitch + col);
}

de::CPf* decx::_MatrixArray::ptr_cpl32(const uint row, const uint col, const uint _seq)
{
    de::CPf* __ptr = reinterpret_cast<de::CPf*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->pitch + col);
}


de::Half* decx::_MatrixArray::ptr_fp16(const uint row, const uint col, const uint _seq)
{
    de::Half* __ptr = reinterpret_cast<de::Half*>(this->MatptrArr.ptr[_seq]);
    return (__ptr + (size_t)row * this->pitch + col);
}


namespace de
{
    _DECX_API_
    de::MatrixArray& CreateMatrixArrayRef();

    _DECX_API_
    de::MatrixArray* CreateMatrixArrayPtr();

    _DECX_API_
    de::MatrixArray& CreateMatrixArrayRef(const int _type, uint width, uint height, uint MatrixNum, const int flag);

    _DECX_API_
    de::MatrixArray* CreateMatrixArrayPtr(const int _type, uint width, uint height, uint MatrixNum, const int flag);
}




de::MatrixArray& de::CreateMatrixArrayRef()
{
    return *(new decx::_MatrixArray());
}



de::MatrixArray* de::CreateMatrixArrayPtr()
{
    return new decx::_MatrixArray();
}




de::MatrixArray& de::CreateMatrixArrayRef(const int _type, uint width, uint height, uint MatrixNum, const int flag)
{
    return *(new decx::_MatrixArray(_type, width, height, MatrixNum, flag));
}




de::MatrixArray* de::CreateMatrixArrayPtr(const int _type, uint width, uint height, uint MatrixNum, const int flag)
{
    return new decx::_MatrixArray(_type, width, height, MatrixNum, flag);
}




void decx::_MatrixArray::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->MatArr);
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->MatArr);
        break;
#endif

    default:
        break;
    }

    decx::alloc::_host_virtual_page_dealloc(&this->MatptrArr);
}



int decx::_MatrixArray::Type()
{
    return this->type;
}



de::MatrixArray& decx::_MatrixArray::SoftCopy(de::MatrixArray& src)
{
    const decx::_MatrixArray& ref_src = dynamic_cast<decx::_MatrixArray&>(src);

    this->MatArr.block = ref_src.MatArr.block;

    this->_attribute_assign(ref_src.type, ref_src.width, ref_src.height, ref_src.ArrayNumber, ref_src._store_type);

    switch (ref_src._store_type)
    {
#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->MatArr);
        break;
#endif

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->MatArr);
        break;

    default:
        break;
    }

    return *this;
}