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
#include "store_types.h"


namespace de
{

    class _DECX_API_ GPU_TensorArray
    {
    public:
        GPU_TensorArray() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint Depth() = 0;


        virtual uint TensorNum() = 0;


        virtual int Type() = 0;



        virtual de::GPU_TensorArray& operator=(de::GPU_TensorArray& src) = 0;


        virtual void release() = 0;
    };


    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef();



    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr();



    _DECX_API_ de::GPU_TensorArray& CreateGPUTensorArrayRef(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num);



    _DECX_API_ de::GPU_TensorArray* CreateGPUTensorArrayPtr(const int _type, const uint width, const uint height, const uint depth, const uint tensor_num);
}


#endif