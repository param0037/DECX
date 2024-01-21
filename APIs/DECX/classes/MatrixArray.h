/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#pragma once

#ifndef _MATRIXARRAY_H_
#define _MATRIXARRAY_H_

#include "../basic.h"

namespace de
{
	/*
	* This class is for matrices array, the matrices included share the same sizes, and store
	* one by one in the memory block, without gap; Compared with de::Tensor<T>, the channel "z"
	* is separated.
	*/
    class _DECX_API_ MatrixArray
    {
    public:
        uint ArrayNumber;


        virtual uint Width() const = 0;


        virtual uint Height() const = 0;


        virtual uint MatrixNumber() const = 0;


        virtual float* ptr_fp32(const uint row, const uint col, const uint _seq) = 0;
        virtual double* ptr_fp64(const uint row, const uint col, const uint _seq) = 0;
        virtual int* ptr_int32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::CPf* ptr_cpl32(const uint row, const uint col, const uint _seq) = 0;
        virtual de::Half* ptr_fp16(const uint row, const uint col, const uint _seq) = 0;
        virtual uint8_t* ptr_uint8(const uint row, const uint col, const uint _seq) = 0;


        virtual de::MatrixArray& SoftCopy(de::MatrixArray& src) = 0;


        virtual void release() = 0;


        virtual de::_DATA_TYPES_FLAGS_ Type() const = 0;


        virtual void Reinterpret(const de::_DATA_TYPES_FLAGS_ _new_type) = 0;
    };


	_DECX_API_
    de::MatrixArray& CreateMatrixArrayRef();

    _DECX_API_
    de::MatrixArray* CreateMatrixArrayPtr();

    _DECX_API_
    de::MatrixArray& CreateMatrixArrayRef(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum);

    _DECX_API_
    de::MatrixArray* CreateMatrixArrayPtr(const de::_DATA_TYPES_FLAGS_ _type, uint width, uint height, uint MatrixNum);
}

#endif