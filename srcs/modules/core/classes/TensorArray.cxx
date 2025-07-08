/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/

#include <Classes/TensorArray.h>
#define MODULE_TAG "TensorArray"
#include <log_console.h>


void decx::_TensorArray::_attribute_assign(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->type = _type;

    this->tensor_num = _tensor_num;
    this->_init = (_type != de::_DATA_TYPES_FLAGS_::_VOID_);

    this->_layout._attribute_assign(_type, _width, _height, _depth);

    this->_gap = this->_layout.dp_x_wp * static_cast<uint64_t>(this->_layout.height);

    this->element_num = static_cast<uint64_t>(this->_layout.depth) * this->_layout.plane[0] * static_cast<uint64_t>(this->tensor_num);
    this->_element_num = static_cast<uint64_t>(this->tensor_num) * this->_gap;
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
    if (this->TensArr.Allocate(this->total_bytes, PAGABLE)){
        DECX_LOG_ERR("Fail to allocate memory for TensorArray on host");
    }

    if (this->TensArr.Allocate(this->tensor_num * sizeof(void*), PAGABLE)){
        return;
    }

    this->TensptrArr[0] = this->TensArr.GetRawPtr();
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr[i] = (uint8_t*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size;
    }
}



void decx::_TensorArray::re_alloc_data_space()
{
    if (this->TensArr.Reallocate(this->total_bytes)) {
        DECX_LOG_ERR("Fail to allocate memory for TensorArray on host");
    }

    if (this->TensptrArr.Reallocate(this->tensor_num * sizeof(void*))) {
        DECX_LOG_ERR("Fail to allocate memory for TensorArray on host\n");
        return;
    }
    
    this->TensptrArr[0] = this->TensArr.GetRawPtr();
    for (uint i = 1; i < this->tensor_num; ++i) {
        this->TensptrArr[i] = (uint8_t*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size;
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
        const uint64_t pre_size = this->total_bytes;

        this->_attribute_assign(_type, _width, _height, _depth, _tensor_num);

        if (this->total_bytes > pre_size) 
        {
            this->TensptrArr.Free();
            this->TensArr.Free();

            this->alloc_data_space();
        }
        else {
            this->TensptrArr[0] = this->TensArr.GetRawPtr();
            for (uint i = 1; i < this->tensor_num; ++i) {
                this->TensptrArr[i] = (void*)((uchar*)this->TensptrArr[i - 1] + this->_gap * this->_layout._single_element_size);
            }
        }
    }
}



decx::_TensorArray::_TensorArray()
{
    this->_exp_data_ptr = this->TensptrArr.GetRawPtr();
    this->_exp_tensor_dscr = &this->_layout;

    this->_attribute_assign(de::_DATA_TYPES_FLAGS_::_VOID_, 0, 0, 0, 0);
}


decx::_TensorArray::_TensorArray(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint _depth, const uint _tensor_num)
{
    this->_exp_data_ptr = this->TensptrArr.GetRawPtr();
    this->_exp_tensor_dscr = &this->_layout;
    
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


de::TensorArray& decx::_TensorArray::SoftCopy(de::TensorArray& src)
{
    decx::_TensorArray& ref_src = dynamic_cast<decx::_TensorArray&>(src);

    this->_attribute_assign(ref_src.type, ref_src._layout.width, ref_src._layout.height, ref_src._layout.depth, ref_src.tensor_num);

    this->TensArr.AllocateRef();

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



void decx::_TensorArray::Extract_SoftCopy(const uint32_t index, de::Tensor& dst) const
{
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    if (index > this->TensorNum() - 1) {
        DecxAssignLastHandle(DecxErrorTypes_e::DECX_FAIL_DimsNotMatching, "Overrange");
        return;
    }

    _dst->_attribute_assign(this->type, this->Width(), this->Height(), this->Depth());
    // _dst->Tens.ptr = this->TensptrArr.ptr[index];
    return;
}


void decx::_TensorArray::release()
{
    this->TensArr.Free();
    this->TensptrArr.Free();
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
