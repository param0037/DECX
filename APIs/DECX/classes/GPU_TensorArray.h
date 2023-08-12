/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/



#ifndef _GPU_TENSORARRAY_H_
#define _GPU_TENSORARRAY_H_


#include "TensorArray.h"


namespace de
{

    class _DECX_API_ GPU_TensorArray
    {
    public:
        GPU_TensorArray() {}


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint Depth() const = 0;


        virtual uint TensorNum() const = 0;


        virtual int Type() const = 0;


        virtual de::GPU_TensorArray& SoftCopy(const de::GPU_TensorArray& src) = 0;


        virtual void release() = 0;
    };


    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef();



    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr();



    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num);



    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num);
}


#endif