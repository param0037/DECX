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

#include "Tensor.h"

namespace de
{
    class _DECX_API_ GPU_Tensor
    {
    public:
        GPU_Tensor() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual de::GPU_Tensor& SoftCopy(de::GPU_Tensor& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;


        virtual de::DH Extract_SoftCopy(const uint32_t index, de::GPU_Tensor& dst) const = 0;
    };


    _DECX_API_ de::GPU_Tensor* CreateGPUTensorPtr();


    _DECX_API_ de::GPU_Tensor& CreateGPUTensorRef();


    _DECX_API_ de::GPU_Tensor* CreateGPUTensorPtr(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint32_t _depth);


    _DECX_API_ de::GPU_Tensor& CreateGPUTensorRef(const de::_DATA_TYPES_FLAGS_ _type, const uint _width, const uint _height, const uint32_t _depth);


    namespace cuda
    {
        _DECX_API_ de::DH PinMemory(de::Tensor& src);


        _DECX_API_ de::DH UnpinMemory(de::Tensor& src);
    }
}

#endif