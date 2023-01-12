/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "../basic.h"
#include "class_utils.h"


/**
* in host, allocate page-locaked memory in 8-times both on width and height
* ensure the utilization of __m128 and __m256, as well as multi threads
*/
namespace de 
{
    class _DECX_API_ Matrix
    {
    public:
        Matrix() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;


        virtual size_t TotalBytes() = 0;


        /* return the reference of the element in the matrix, which locates on specific row and colume
        * \params row -> where the element locates on row
        * \params col -> where the element locates on colume
        */
        virtual float* ptr_fp32(const int row, const int col) = 0;
        virtual double* ptr_fp64(const int row, const int col) = 0;
        virtual int* ptr_int32(const int row, const int col) = 0;
        virtual de::CPf* ptr_cpl32(const int row, const int col) = 0;
        virtual de::Half* ptr_fp16(const int row, const int col) = 0;
        virtual uint8_t* ptr_uint8(const int row, const int col) = 0;


        virtual void release() = 0;


        virtual de::Matrix& operator=(de::Matrix& src) = 0;


        virtual int Type() = 0;


        ~Matrix() {}
    };


    _DECX_API_ de::Matrix* CreateMatrixPtr();


    _DECX_API_ de::Matrix& CreateMatrixRef();


    _DECX_API_ de::Matrix* CreateMatrixPtr(const int type, const uint _width, const uint _height, const int store_type);


    _DECX_API_ de::Matrix& CreateMatrixRef(const int type, const uint _width, const uint _height, const int store_type);
}

#endif