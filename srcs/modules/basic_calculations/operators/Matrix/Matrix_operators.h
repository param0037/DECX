/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _MATRIX_OPERATORS_H_
#define _MATRIX_OPERATORS_H_

#ifdef _DECX_CUDA_PARTS_
#include "../../../classes/GPU_Matrix.h"
#endif
#include "../../../classes/Matrix.h"


namespace de
{
    namespace cpu 
    {
        _DECX_API_ de::DH Add(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH Add(de::Matrix& src, void* __x, de::Matrix& dst);


        _DECX_API_ de::DH Div(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH Div(de::Matrix& src, void* __x, de::Matrix& dst);


        _DECX_API_ de::DH Div(void* __x, de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Fma(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


        _DECX_API_ de::DH Fma(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH Fms(de::Matrix& A, de::Matrix& B, de::Matrix& C, de::Matrix& dst);


        _DECX_API_ de::DH Fms(de::Matrix& src, void* __x, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH Mul(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH Mul(de::Matrix& src, void* __x, de::Matrix& dst);


        _DECX_API_ de::DH Sub(de::Matrix& A, de::Matrix& B, de::Matrix& dst);


        _DECX_API_ de::DH Sub(de::Matrix& src, void* __x, de::Matrix& dst);


        _DECX_API_ de::DH Sub(void* __x, de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Clip(de::Matrix& src, de::Matrix& dst, const de::Point2D_d range);
    }

#ifdef _DECX_CUDA_PARTS_
    namespace cuda
    {
        _DECX_API_  de::DH Add(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Add(de::GPU_Matrix& src, void* __x, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Div(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Div(de::GPU_Matrix& src, void* __x, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Div(void* __x, de::GPU_Matrix& src, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Fma(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Fma(de::GPU_Matrix& src, void* __x, de::GPU_Matrix& B, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Fms(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& C, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Fms(de::GPU_Matrix& src, void* __x, de::GPU_Matrix& B, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Mul(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Mul(de::GPU_Matrix& src, void* __x, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Sub(de::GPU_Matrix& A, de::GPU_Matrix& B, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Sub(de::GPU_Matrix& src, void* __x, de::GPU_Matrix& dst);


        _DECX_API_  de::DH Sub(void* __x, de::GPU_Matrix& src, de::GPU_Matrix& dst);
    }
#endif
}



#endif