/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _Tensor_H_
#define _Tensor_H_

#include "../basic.h"
#include "../vectorial/vector4.h"


namespace de
{
    class _DECX_API_ Tensor
    {
    public:
        Tensor() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual float* ptr_fp32(const int x, const int y, const int z) = 0;
        virtual int* ptr_int32(const int x, const int y, const int z) = 0;
        virtual double* ptr_fp64(const int x, const int y, const int z) = 0;
        virtual de::Half* ptr_fp16(const int x, const int y, const int z) = 0;
        virtual de::CPf* ptr_cpl32(const int x, const int y, const int z) = 0;
        virtual uint8_t* ptr_uint8(const int x, const int y, const int z) = 0;
        virtual de::Vector4f* ptr_vec4f(const int x, const int y, const int z) = 0;


        virtual de::Tensor& SoftCopy(de::Tensor& src) = 0;


        virtual void release() = 0;


        virtual int Type() const = 0;
    };


    de::Tensor* CreateTensorPtr();


    de::Tensor& CreateTensorRef();


    de::Tensor* CreateTensorPtr(const int _type, const uint _width, const uint _height, const uint _depth, const int flag);


    de::Tensor& CreateTensorRef(const int _type, const uint _width, const uint _height, const uint _depth, const int flag);
}

#endif