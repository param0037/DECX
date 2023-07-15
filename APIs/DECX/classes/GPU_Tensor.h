/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GPU_TENSOR_H_
#define _GPU_TENSOR_H_

#include "tensor.h"
#include "store_types.h"


namespace de
{
    class _DECX_API_ GPU_Tensor
    {
    public:
        GPU_Tensor() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual de::GPU_Tensor& operator=(de::GPU_Tensor& src) = 0;


        virtual void release() = 0;


        virtual int Type() = 0;
    };


    _DECX_API_ de::GPU_Tensor* CreateGPUTensorPtr();


    _DECX_API_ de::GPU_Tensor& CreateGPUTensorRef();


    _DECX_API_ de::GPU_Tensor* CreateGPUTensorPtr(const int _type, const uint _width, const uint _height, const uint _depth);


    _DECX_API_ de::GPU_Tensor& CreateGPUTensorRef(const int _type, const uint _width, const uint _height, const uint _depth);
}

#endif