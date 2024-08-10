/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _TENSORARRAY_H_
#define _TENSORARRAY_H_

#include "../basic.h"

#ifdef __cplusplus
namespace de
{
    class _DECX_API_ TensorArray
    {
    protected:
        _SHADOW_ATTRIBUTE_(void**) _exp_data_ptr;
        _SHADOW_ATTRIBUTE_(de::TensorLayout) _exp_tensor_dscr;

    public:
        TensorArray() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual uint TensorNum() const = 0;


        /*virtual float* ptr_fp32(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual int* ptr_int32(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual de::Half* ptr_fp16(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual double* ptr_fp64(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual uint8_t* ptr_uint8(const int x, const int y, const int z, const int tensor_id) = 0;*/


        template <typename _ptr_type>
        _ptr_type* ptr(const int x, const int y, const int z, const int tensor_id)
        {
            return ((_ptr_type*)(*this->_exp_data_ptr)[tensor_id]) + x * this->_exp_tensor_dscr->dp_x_wp + y * this->_exp_tensor_dscr->dpitch + z;
        }


        virtual de::TensorArray& SoftCopy(de::TensorArray& src) = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::DH Extract_SoftCopy(const uint32_t index, de::Tensor& dst) const = 0;


        virtual void release() = 0;
    };
}


namespace de
{
    _DECX_API_ de::TensorArray& CreateTensorArrayRef();


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr();


    _DECX_API_ de::TensorArray& CreateTensorArrayRef(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num);


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint width, const uint height, const uint depth, const uint tensor_num);


    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::TensorArray& src);


        _DECX_API_ de::DH UnpinMemory(de::TensorArray& src);
    }
}
#endif


#ifdef _C_CONTEXT_
typedef struct DECX_TensorArray_t
{
    void* _segment;
}DECX_TensorArray;


_DECX_API_ DECX_TensorArray DE_CreateEmptyTensorArray();


_DECX_API_ DECX_TensorArray DE_CreateTensorArray(const int8_t type, const uint32_t _width, const uint32_t _height,
    const uint32_t _depth, const uint32_t tensor_num);
#endif


#endif