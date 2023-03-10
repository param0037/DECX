/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#pragma once

#ifndef _GPU_MATRIXARRAY_H_
#define _GPU_MATRIXARRAY_H_

#include "../basic.h"
#include "MatrixArray.h"



namespace de
{
    class _DECX_API_ GPU_MatrixArray
    {
    public:
        uint ArrayNumber();


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual uint MatrixNumber() = 0;


        virtual void Load_from_host(de::MatrixArray& src) = 0;


        virtual void Load_to_host(de::MatrixArray& dst) = 0;


        virtual de::GPU_MatrixArray& operator=(de::GPU_MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual int Type() = 0;
    };


    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef();


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr();


    _DECX_API_
    de::GPU_MatrixArray& CreateGPUMatrixArrayRef(const int _type, const uint _width, const uint _height, const uint _Mat_number);


    _DECX_API_
    de::GPU_MatrixArray* CreateGPUMatrixArrayPtr(const int _type, const uint _width, const uint _height, const uint _Mat_number);
}


#endif