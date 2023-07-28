/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "TensorArray.h"


void decx::_TensorArray::_attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    /*this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;
    this->tensor_num = _tensor_num;

    this->TensArr.ptr = NULL;           this->TensArr.block = NULL;
    this->TensptrArr.ptr = NULL;        this->TensptrArr.block = NULL;

    this->_store_type = store_type;
    if (this->_single_element_size == 2) {
        this->wpitch = decx::utils::ceil<uint>(_width, 8) * 8;
    }
    else {
        this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;
    }

    uint _alignment = 0;
    switch (this->_single_element_size)
    {
    case 4:
        _alignment = _TENSOR_ALIGN_DEPTH_4B_;     break;
    case 8:
        _alignment = _TENSOR_ALIGN_DEPTH_8B_;     break;
    case 2:
        _alignment = _TENSOR_ALIGN_DEPTH_2B_;     break;
    case 1:
        _alignment = _TENSOR_ALIGN_DEPTH_1B_;     break;
    default:
        break;
    }

    this->dpitch = decx::utils::ceil<uint>(_depth, _alignment) * _alignment;
    
    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->_gap = this->dp_x_wp * static_cast<size_t>(this->height);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * this->_single_element_size;*/

    this->type = _type;

    this->type = _type;
    this->tensor_num = _tensor_num;
    this->_init = true;

    this->_store_type = store_type;

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->_gap = this->_layout.dp_x_wp * static_cast<size_t>(this->_layout.height);

    this->element_num = static_cast<size_t>(this->_layout.depth) * this->_layout.plane[0] * static_cast<size_t>(this->tensor_num);
    this->_element_num = static_cast<size_t>(this->tensor_num) * this->_gap;
    this->total_bytes = this->_element_num * this->_layout._single_element_size;
}




uint decx::_TensorArray::Width() const
{
    return this->_layout.width;
}


uint decx::_TensorArray::Height() const
{
    return this->_layout.height;
}


uint decx::_TensorArray::Depth() const
{
    return this->_layout.depth;
}


uint decx::_TensorArray::TensorNum() const
{
    return this->tensor_num;
}


void decx::_TensorArray::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<void>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_malloc<void>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    default:
        break;
    }

    memset(this->TensArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_malloc<void*>(&this->TensptrArr, this->tensor_num * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = (void*)((uchar*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_layout._single_element_size);
    }
}



void decx::_TensorArray::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_realloc<void>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        if (decx::alloc::_host_virtual_page_realloc<void>(&this->TensArr, this->total_bytes)) {
            Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
            exit(-1);
        }
        break;

    default:
        break;
    }

    memset(this->TensArr.ptr, 0, this->total_bytes);

    if (decx::alloc::_host_virtual_page_realloc<void*>(&this->TensptrArr, this->tensor_num * sizeof(void*))) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        return;
    }
    this->TensptrArr.ptr[0] = this->TensArr.ptr;
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr.ptr[i] = (void*)((uchar*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_layout._single_element_size);
    }
}




void decx::_TensorArray::construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num, store_type);

    this->alloc_data_space();
}




void decx::_TensorArray::re_construct(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    if (this->type != _type || this->_layout.width != _width || this->_layout.height != _height || this->_layout.depth != _depth ||
        this->tensor_num != _tensor_num || this->_store_type != store_type) 
    {
        const size_t pre_size = this->total_bytes;
        const int pre_store_type = this->_store_type;

        this->_attribute_assign(_type, _width, _height, _depth, _tensor_num, store_type);

        if (this->total_bytes > pre_size || pre_store_type != store_type) 
        {
            decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);

            if (pre_store_type != store_type) {
                switch (pre_store_type)
                {
                case decx::DATA_STORE_TYPE::Page_Default:
                    decx::alloc::_host_virtual_page_dealloc(&this->TensArr);
                    break;

                case decx::DATA_STORE_TYPE::Page_Locked:
                    decx::alloc::_host_fixed_page_dealloc(&this->TensArr);
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
        else {
            this->TensptrArr.ptr[0] = this->TensArr.ptr;
            for (uint i = 1; i < this->tensor_num; ++i) {
                this->TensptrArr.ptr[i] = (void*)((uchar*)this->TensptrArr.ptr[i - 1] + this->_gap * this->_layout._single_element_size);
            }
        }
    }
}



decx::_TensorArray::_TensorArray()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0, 0);
}




decx::_TensorArray::_TensorArray(const int _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num, const int store_type)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num, store_type);

    this->alloc_data_space();
}


namespace de
{
    _DECX_API_ de::TensorArray& CreateTensorArrayRef();


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr();


    _DECX_API_ de::TensorArray& CreateTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);
}



de::TensorArray& de::CreateTensorArrayRef()
{
    return *(new decx::_TensorArray());
}




de::TensorArray* de::CreateTensorArrayPtr()
{
    return new decx::_TensorArray();
}




de::TensorArray& de::CreateTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type)
{
    return *(new decx::_TensorArray(_type, width, height, depth, tensor_num, store_type));
}




de::TensorArray* de::CreateTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type)
{
    return new decx::_TensorArray(_type, width, height, depth, tensor_num, store_type);
}




float* decx::_TensorArray::ptr_fp32(const int x, const int y, const int z, const int tensor_id)
{
    float* _ptr = reinterpret_cast<float*>(this->TensptrArr.ptr[tensor_id]);
    return (_ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z));
}


int* decx::_TensorArray::ptr_int32(const int x, const int y, const int z, const int tensor_id)
{
    int* _ptr = reinterpret_cast<int*>(this->TensptrArr.ptr[tensor_id]);
    return (_ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z));
}



de::Half* decx::_TensorArray::ptr_fp16(const int x, const int y, const int z, const int tensor_id)
{
    de::Half* _ptr = reinterpret_cast<de::Half*>(this->TensptrArr.ptr[tensor_id]);
    return (_ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z));
}



double* decx::_TensorArray::ptr_fp64(const int x, const int y, const int z, const int tensor_id)
{
    double* _ptr = reinterpret_cast<double*>(this->TensptrArr.ptr[tensor_id]);
    return (_ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z));
}



uint8_t* decx::_TensorArray::ptr_uint8(const int x, const int y, const int z, const int tensor_id)
{
    uint8_t* _ptr = reinterpret_cast<uint8_t*>(this->TensptrArr.ptr[tensor_id]);
    return (_ptr +
        ((size_t)x * this->_layout.dp_x_wp + (size_t)y * (size_t)this->_layout.dpitch + (size_t)z));
}



de::TensorArray& decx::_TensorArray::SoftCopy(de::TensorArray& src)
{
    decx::_TensorArray& ref_src = dynamic_cast<decx::_TensorArray&>(src);

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth, ref_src.tensor_num, ref_src._store_type);

    switch (ref_src._store_type)
    {
    

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->TensArr);
        break;
#endif

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->TensArr);
        break;

    default:
        break;
    }

    return *this;
}


int decx::_TensorArray::Type() const
{
    return this->type;
}


void decx::_TensorArray::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->TensArr);
        break;

#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->TensArr);
        break;
#endif

    default:
        break;
    }

    decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);
}


int decx::_TensorArray::get_store_type() const
{
    return this->_store_type;
}


const decx::_tensor_layout& decx::_TensorArray::get_layout() const
{
    return this->_layout;
}


bool decx::_TensorArray::is_init() const
{
    return this->_init;
}


uint64_t decx::_TensorArray::get_total_bytes() const
{
    return this->total_bytes;
}