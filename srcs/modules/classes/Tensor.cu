/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Tensor.h"



decx::_Tensor::_Tensor()
{
    this->_attribute_assign(decx::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0);
    this->_init = false;
}



void decx::_Tensor::_attribute_assign(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->type = _type;
    this->_single_element_size = decx::core::_size_mapping(_type);

    this->width = _width;
    this->height = _height;
    this->depth = _depth;

    this->Tens.ptr = NULL;           this->Tens.block = NULL;

    this->_store_type = store_type;
    this->wpitch = decx::utils::ceil<uint>(_width, 4) * 4;

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
    case 16:
        _alignment = _TENSOR_ALIGN_DEPTH_16B_;    break;
    default:
        break;
    }
    this->dpitch = decx::utils::ceil<uint>(_depth, _alignment) * _alignment;
    this->_init = true;

    this->dp_x_wp = static_cast<size_t>(this->dpitch) * static_cast<size_t>(this->wpitch);

    this->plane[0] = static_cast<size_t>(this->height) * static_cast<size_t>(this->width);
    this->plane[1] = static_cast<size_t>(this->depth) * static_cast<size_t>(this->width);
    this->plane[2] = static_cast<size_t>(this->height) * static_cast<size_t>(this->depth);

    this->element_num = static_cast<size_t>(this->depth) * this->plane[0];
    this->_element_num = static_cast<size_t>(this->height) * this->dp_x_wp;
    this->total_bytes = this->_element_num * sizeof(float);
}




void decx::_Tensor::alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        if (decx::alloc::_host_fixed_page_malloc<void>(&this->Tens, this->total_bytes)) {
            Print_Error_Message(4, "Tensor malloc failed! Please check if there is enough space in your RAM.");
            exit(-1);
        }
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc<void>(&this->Tens, this->total_bytes);
        break;

    default:
        break;
    }

    memset(this->Tens.ptr, 0, this->total_bytes);
}




void decx::_Tensor::re_alloc_data_space()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_realloc<void>(&this->Tens, this->total_bytes);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_realloc<void>(&this->Tens, this->total_bytes);
        break;

    default:
        break;
    }

    memset(this->Tens.ptr, 0, this->total_bytes);
}



void decx::_Tensor::construct(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->_attribute_assign(_type, _width, _height, _depth, store_type);

    this->alloc_data_space();
}




void decx::_Tensor::re_construct(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type)
{
    /*const size_t pre_size = this->total_bytes;
    const int pre_store_type = this->_store_type;
    this->_attribute_assign(_type, _width, _height, _depth, store_type);

    if (this->total_bytes > pre_size || store_type != pre_store_type) {
        this->re_alloc_data_space();
    }*/

    // If all the parameters are the same, it is meaningless to re-construt the data
    if (this->type != _type || this->width != _width || this->height != _height || this->_store_type != store_type)
    {
        const size_t pre_size = this->total_bytes;
        const int pre_store_type = this->_store_type;

        this->_attribute_assign(_type, _width, _height, _depth, store_type);

        if (this->total_bytes > pre_size || pre_store_type != store_type) {
            // deallocate according to the current memory pool first
            if (pre_store_type != store_type && this->Tens.ptr == NULL) {
                switch (pre_store_type)
                {
                case decx::DATA_STORE_TYPE::Page_Default:
                    decx::alloc::_host_virtual_page_dealloc(&this->Tens);
                    break;

                case decx::DATA_STORE_TYPE::Page_Locked:
                    decx::alloc::_host_fixed_page_dealloc(&this->Tens);
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




decx::_Tensor::_Tensor(const int _type, const uint _width, const uint _height, const uint _depth, const int store_type)
{
    this->_attribute_assign(_type, _width, _height, _depth, store_type);

    this->alloc_data_space();
}




void decx::_Tensor::release()
{
    switch (this->_store_type)
    {
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_dealloc(&this->Tens);
        break;

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_dealloc(&this->Tens);
        break;

    default:
        break;
    }
}



int decx::_Tensor::Type()
{
    return this->type;
}




float* decx::_Tensor::ptr_fp32(const int x, const int y, const int z)
{
    float* ptr = reinterpret_cast<float*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}


int* decx::_Tensor::ptr_int32(const int x, const int y, const int z)
{
    int* ptr = reinterpret_cast<int*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}


uint8_t* decx::_Tensor::ptr_uint8(const int x, const int y, const int z)
{
    uint8_t* ptr = reinterpret_cast<uint8_t*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}


de::CPf* decx::_Tensor::ptr_cpl32(const int x, const int y, const int z)
{
    de::CPf* ptr = reinterpret_cast<de::CPf*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}


double* decx::_Tensor::ptr_fp64(const int x, const int y, const int z)
{
    double* ptr = reinterpret_cast<double*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}


de::Half* decx::_Tensor::ptr_fp16(const int x, const int y, const int z)
{
    de::Half* ptr = reinterpret_cast<de::Half*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}


de::Vector4f* decx::_Tensor::ptr_vec4f(const int x, const int y, const int z)
{
    de::Vector4f* ptr = reinterpret_cast<de::Vector4f*>(this->Tens.ptr);
    return ptr +
        ((size_t)x * this->dp_x_wp + (size_t)y * (size_t)this->dpitch + (size_t)z);
}



namespace de
{
    _DECX_API_ de::Tensor* CreateTensorPtr();


    _DECX_API_ de::Tensor& CreateTensorRef();


    _DECX_API_ de::Tensor* CreateTensorPtr(const int _type, const uint _width, const uint _height, const uint _depth, const int flag);


    _DECX_API_ de::Tensor& CreateTensorRef(const int _type, const uint _width, const uint _height, const uint _depth, const int flag);
}



_DECX_API_ de::Tensor& de::CreateTensorRef()
{
    return *(new decx::_Tensor());
}



_DECX_API_ de::Tensor* de::CreateTensorPtr()
{
    return new decx::_Tensor();
}





_DECX_API_ de::Tensor& de::CreateTensorRef(const int _type, const uint _width, const uint _height, const uint _depth, const int flag)
{
    return *(new decx::_Tensor(_type, _width, _height, _depth, flag));
}



_DECX_API_ de::Tensor* de::CreateTensorPtr(const int _type, const uint _width, const uint _height, const uint _depth, const int flag)
{
    return new decx::_Tensor(_type, _width, _height, _depth, flag);
}



de::Tensor& decx::_Tensor::SoftCopy(de::Tensor& src)
{
    decx::_Tensor& ref_src = dynamic_cast<decx::_Tensor&>(src);

    this->Tens.block = ref_src.Tens.block;

    this->_attribute_assign(ref_src._store_type, ref_src.width, ref_src.height, ref_src.depth, ref_src._store_type);

    switch (ref_src._store_type)
    {
#ifdef _DECX_CUDA_CODES_
    case decx::DATA_STORE_TYPE::Page_Locked:
        decx::alloc::_host_fixed_page_malloc_same_place(&this->Tens);
        break;
#endif

    case decx::DATA_STORE_TYPE::Page_Default:
        decx::alloc::_host_virtual_page_malloc_same_place(&this->Tens);
        break;

    default:
        break;
    }

    return *this;
}
