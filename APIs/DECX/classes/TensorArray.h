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

namespace de
{
    class _DECX_API_ TensorArray
    {
    public:
        TensorArray() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual uint TensorNum() const = 0;


        virtual float* ptr_fp32(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual int* ptr_int32(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual de::Half* ptr_fp16(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual double* ptr_fp64(const int x, const int y, const int z, const int tensor_id) = 0;
        virtual uint8_t* ptr_uint8(const int x, const int y, const int z, const int tensor_id) = 0;


        virtual de::TensorArray& SoftCopy(de::TensorArray& src) = 0;


        virtual int Type() const = 0;


        virtual void release() = 0;
    };
}


namespace de
{
    _DECX_API_ de::TensorArray& CreateTensorArrayRef();


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr();


    _DECX_API_ de::TensorArray& CreateTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);


    _DECX_API_ de::TensorArray* CreateTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num, const int store_type);

}


#endif