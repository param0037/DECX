/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "GPU_MatrixArray.h"


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
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_single_element_size);
    }
}



void decx::_GPU_MatrixArray::alloc_data_space()
{
    if (decx::alloc::_device_malloc(&this->MatArr, this->total_bytes)) {
        Print_Error_Message(4, "de::GPU_MatrixArray<T>:: Fail to allocate memory\n");
        exit(-1);
    }

    cudaMemset(this->MatArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->MatptrArr, this->ArrayNumber * this->_single_element_size)) {
        Print_Error_Message(4, "Fail to allocate memory for pointer array on host\n");
        return;
    }
    this->MatptrArr.ptr[0] = this->MatArr.ptr;
    for (int i = 1; i < this->ArrayNumber; ++i) {
        this->MatptrArr.ptr[i] = (void*)((uchar*)this->MatptrArr.ptr[i - 1] + this->_plane * this->_single_element_size);
    }
}




void decx::_GPU_MatrixArray::_attribute_assign(const int _type, uint _width, uint _height, uint MatrixNum)
{
    this->type = _type;
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

    this->total_bytes = this->_element_num * this->_single_element_size;
}



void decx::_GPU_MatrixArray::construct(const int _type, uint _width, uint _height, uint MatrixNum)
{
    this->_attribute_assign(_type, _width, _height, MatrixNum);

    this->alloc_data_space();
}



void decx::_GPU_MatrixArray::re_construct(const int _type, uint _width, uint _height, uint MatrixNum)
{
    if (_width != this->width || _height != this->height || MatrixNum != this->ArrayNumber) {
        const size_t pre_size = this->total_bytes;
        this->_attribute_assign(_type, _width, _height, MatrixNum);

        if (this->total_bytes > pre_size) {
            this->re_alloc_data_space();
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



void decx::_GPU_MatrixArray::Load_from_host(de::MatrixArray& src)
{
    decx::_MatrixArray* _src = dynamic_cast<_MatrixArray*>(&src);
    if (_src->type != this->type) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        return;
    }

    for (int i = 0; i < this->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2D(this->MatptrArr.ptr[i], this->pitch * this->_single_element_size,
            _src->MatptrArr.ptr[i], _src->pitch * this->_single_element_size, this->width * this->_single_element_size, this->height,
            cudaMemcpyHostToDevice));
    }
}



void decx::_GPU_MatrixArray::Load_to_host(de::MatrixArray& dst)
{
    decx::_MatrixArray* _dst = dynamic_cast<_MatrixArray*>(&dst);
    if (_dst->type != this->type) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        return;
    }

    for (int i = 0; i < this->ArrayNumber; ++i) {
        checkCudaErrors(cudaMemcpy2D(_dst->MatptrArr.ptr[i], _dst->pitch * this->_single_element_size,
            this->MatptrArr.ptr[i], this->pitch * this->_single_element_size, _dst->width * this->_single_element_size, _dst->height,
            cudaMemcpyDeviceToHost));
    }
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

    this->_attribute_assign(ref_src.type, ref_src.width, ref_src.height, ref_src.ArrayNumber);
    decx::alloc::_device_malloc_same_place(&this->MatArr);

    return *this;
}


int decx::_GPU_MatrixArray::Type()
{
    return this->type;
}
