/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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