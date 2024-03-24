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


void decx::_TensorArray::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->type = _type;

    this->tensor_num = _tensor_num;
    this->_init = (_type != de::_DATA_TYPES_FLAGS_::_VOID_);

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
    if (decx::alloc::_host_virtual_page_malloc<void>(&this->TensArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        exit(-1);
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
    if (decx::alloc::_host_virtual_page_realloc<void>(&this->TensArr, this->total_bytes)) {
        Print_Error_Message(4, "Fail to allocate memory for TensorArray on host\n");
        exit(-1);
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




void decx::_TensorArray::construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}




void decx::_TensorArray::re_construct(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    if (this->type != _type || this->_layout.width != _width || this->_layout.height != _height || this->_layout.depth != _depth ||
        this->tensor_num != _tensor_num) 
    {
        const size_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

        if (this->total_bytes > pre_size) 
        {
            decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);

            decx::alloc::_host_virtual_page_dealloc(&this->TensArr);

            this->alloc_data_space();
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
    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0);
}




decx::_TensorArray::_TensorArray(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

    this->alloc_data_space();
}



#if _CPP_EXPORT_ENABLED_
de::TensorArray& de::CreateTensorArrayRef()
{
    return *(new decx::_TensorArray());
}


de::TensorArray* de::CreateTensorArrayPtr()
{
    return new decx::_TensorArray();
}


de::TensorArray& de::CreateTensorArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num)
{
    return *(new decx::_TensorArray(_type, width, height, depth, tensor_num));
}


de::TensorArray* de::CreateTensorArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num)
{
    return new decx::_TensorArray(_type, width, height, depth, tensor_num);
}
#endif



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

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth, ref_src.tensor_num);

    decx::alloc::_host_virtual_page_malloc_same_place(&this->TensArr);

    return *this;
}


de::_DATA_TYPES_FLAGS_ decx::_TensorArray::Type() const
{
    return this->type;
}


void decx::_TensorArray::Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type)
{
    this->type = _new_type;
}



de::DH decx::_TensorArray::Extract_SoftCopy(const uint32_t index, de::Tensor& dst) const
{
    de::DH handle;
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    if (index > this->TensorNum() - 1) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            "Overrange\n");
        return handle;
    }

    _dst->_attribute_assign(this->type, this->Width(), this->Height(), this->Depth());
    _dst->Tens.ptr = this->TensptrArr.ptr[index];

    return handle;
}


void decx::_TensorArray::release()
{
    decx::alloc::_host_virtual_page_dealloc(&this->TensArr);

    decx::alloc::_host_virtual_page_dealloc(&this->TensptrArr);
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



#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    _DECX_API_ DECX_TensorArray DE_CreateEmptyTensorArray()
    {
        return DECX_TensorArray(new decx::_TensorArray());
    }


    _DECX_API_ DECX_TensorArray DE_CreateTensorArray(const int8_t type, const uint32_t width, const uint32_t height,
        const uint32_t depth, const uint32_t tensor_num)
    {
        return DECX_TensorArray(new decx::_TensorArray(static_cast<de::_DATA_TYPES_FLAGS_>(type), width, height, depth, tensor_num));
    }
#ifdef __cplusplus
}
#endif
#endif
