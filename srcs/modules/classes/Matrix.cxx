/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Matrix.h"


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


namespace de
{
    _DECX_API_ de::Matrix* CreateMatrixPtr();


    _DECX_API_ de::Matrix& CreateMatrixRef();


    _DECX_API_ de::Matrix* CreateMatrixPtr(const int type, const uint _width, const uint _height, const int store_type);


    _DECX_API_ de::Matrix& CreateMatrixRef(const int type, const uint _width, const uint _height, const int store_type);
}



void decx::_Matrix::alloc_data_space()
{
    switch (this->Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<void>(&this->Mat, this->total_bytes)) {
            Print_Error_Message(4, ALLOC_FAIL);
            return;
        }
        break;
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<void>(&this->Mat, this->total_bytes)) {
            Print_Error_Message(4, ALLOC_FAIL);
            return;
        }
        break;
    default:
        Print_Error_Message(4, MEANINGLESS_FLAG);
        break;
    }
}



void decx::_Matrix::re_alloc_data_space()
{
    switch (this->Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<void>(&this->Mat, this->total_bytes)) {
            Print_Error_Message(4, ALLOC_FAIL);
            return;
        }
        break;
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<void>(&this->Mat, this->total_bytes)) {
            Print_Error_Message(4, ALLOC_FAIL);
            return;
        }
        break;
    default:
        Print_Error_Message(4, MEANINGLESS_FLAG);
        break;
    }
}


int decx::_Matrix::Type() const
{
    return this->type;
}


uint32_t decx::_Matrix::Pitch() const
{
    return this->_layout.pitch;
}


const decx::_matrix_layout& decx::_Matrix::get_layout() const
{
    return this->_layout;
}


void decx::_Matrix::construct(const int type, uint _width, uint _height, const int flag)
{
    this->_attribute_assign(type, _width, _height, flag);

    this->alloc_data_space();
}


// 当free掉page_locked的内存区域并重新分配page_defualt的时候会出bug
void decx::_Matrix::re_construct(const int type, uint _width, uint _height, const int flag)
{
    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != type || this->_layout.width != _width || this->_layout.height != _height || this->Store_Type != flag)
    {
        const size_t pre_size = this->total_bytes;
        const int pre_store_type = this->Store_Type;

        this->_attribute_assign(type, _width, _height, flag);
        
        if (this->total_bytes > pre_size || pre_store_type != flag) {
            // deallocate according to the current memory pool first
            if (pre_store_type != flag && this->Mat.ptr == NULL) {
                switch (pre_store_type)
                {
                case decx::DATA_STORE_TYPE::Page_Default:
                    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
                    break;

                case decx::DATA_STORE_TYPE::Page_Locked:
                    decx::alloc::_host_fixed_page_dealloc(&this->Mat);
                    break;
                default:
                    break;
                }
                this->alloc_data_space();
            }
            else {
                this->re_alloc_data_space();
            }
        }
    }
}



void decx::_Matrix::_attribute_assign(const int _type, const uint _width, const uint _height, const int store_type)
{
    this->type = _type;

    this->_layout._attribute_assign(_type, _width, _height);

    this->_init = true;

    this->Store_Type = store_type;

    if (_type != decx::_DATA_TYPES_FLAGS_::_VOID_)
    {
        this->_init = true;

        uint64_t element_num = static_cast<size_t>(this->_layout.pitch) * static_cast<size_t>(_height);
        this->total_bytes = (element_num) * this->_layout._single_element_size;
    }
}




void decx::_Matrix::release()
{
    switch (this->Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->Mat);
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->Mat);
        break;
#endif

    default:
        break;
    }
}


decx::_Matrix::_Matrix()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0);
    this->_init = false;
}



decx::_Matrix::_Matrix(const int type, const uint _width, const uint _height, const int store_type)
{
    this->construct(type, _width, _height, store_type);
}


uint32_t decx::_Matrix::Width() const
{
    return this->_layout.width;
}


uint32_t decx::_Matrix::Height() const
{
    return this->_layout.height;
}


float* decx::_Matrix::ptr_fp32(const int row, const int col)
{
    float* __ptr = reinterpret_cast<float*>(this->Mat.ptr);
    return __ptr + this->_layout.pitch * (size_t)row + (size_t)col;
}


double* decx::_Matrix::ptr_fp64(const int row, const int col)
{
    double* __ptr = reinterpret_cast<double*>(this->Mat.ptr);
    return __ptr + this->_layout.pitch * (size_t)row + (size_t)col;
}


int* decx::_Matrix::ptr_int32(const int row, const int col)
{
    int* __ptr = reinterpret_cast<int*>(this->Mat.ptr);
    return __ptr + this->_layout.pitch * (size_t)row + (size_t)col;
}


de::CPf* decx::_Matrix::ptr_cpl32(const int row, const int col)
{
    de::CPf* __ptr = reinterpret_cast<de::CPf*>(this->Mat.ptr);
    return __ptr + this->_layout.pitch * (size_t)row + (size_t)col;
}


de::Half* decx::_Matrix::ptr_fp16(const int row, const int col)
{
    de::Half* __ptr = reinterpret_cast<de::Half*>(this->Mat.ptr);
    return __ptr + this->_layout.pitch * (size_t)row + (size_t)col;
}


uint8_t* decx::_Matrix::ptr_uint8(const int row, const int col)
{
    uint8_t* __ptr = reinterpret_cast<uint8_t*>(this->Mat.ptr);
    return __ptr + this->_layout.pitch * (size_t)row + (size_t)col;
}


decx::_Matrix::~_Matrix()
{
    if (this->Mat.ptr != NULL) {
        this->release();
    }
}



int32_t decx::_Matrix::get_store_type() const
{
    return this->Store_Type;
}


bool decx::_Matrix::is_init() const
{
    return this->_init;
}


uint64_t decx::_Matrix::get_total_bytes() const
{
    return this->total_bytes;
}


// Matrix and Tensor creation


de::Matrix& de::CreateMatrixRef()
{
    return *(new decx::_Matrix());
}


de::Matrix* de::CreateMatrixPtr()
{
    return new decx::_Matrix();
}



de::Matrix& de::CreateMatrixRef(const int type, const uint _width, const uint _height, const int store_type)
{
    return *(new decx::_Matrix(type, _width, _height, store_type));
}



de::Matrix* de::CreateMatrixPtr(const int type, const uint _width, const uint _height, const int store_type)
{
    return new decx::_Matrix(type, _width, _height, store_type);
}



de::Matrix& decx::_Matrix::SoftCopy(de::Matrix& src)
{
    const decx::_Matrix& ref_src = dynamic_cast<decx::_Matrix&>(src);

    this->Mat.block = ref_src.Mat.block;

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src.Store_Type);

    switch (ref_src.Store_Type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->Mat);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->Mat);
        break;
    }

    return *this;
}