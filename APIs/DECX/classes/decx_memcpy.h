/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DECX_MEMCPY_H_
#define _DECX_MEMCPY_H_

#include "class_utils.h"
#include "Matrix.h"
#include "GPU_Matrix.h"
#include "Vector.h"
#include "GPU_Vector.h"
#include "Tensor.h"
#include "GPU_Tensor.h"
#include "MatrixArray.h"
#include "GPU_MatrixArray.h"
#include "TensorArray.h"
#include "GPU_TensorArray.h"
#include "../Async/DecxStream.h"



namespace de
{
    enum DECX_Memcpy_Flags
    {
        DECX_MEMCPY_H2D = 0,
        DECX_MEMCPY_D2H = 1,
    };
}



namespace de
{
    _DECX_API_ de::DH Memcpy(de::Vector& __host, de::GPU_Vector& __device, const size_t start, const size_t cpy_size,
        const int _memcpy_flag);
    _DECX_API_ de::DH Memcpy_Async(de::Vector& __host, de::GPU_Vector& __device, const size_t start, const size_t cpy_size,
        const int _memcpy_flag, de::DecxStream& S);

    // For the matrix, the data is copied to the very beginning of the destinated matrix
    _DECX_API_ de::DH Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
        const int _memcpy_flag);
    _DECX_API_ de::DH Memcpy_Async(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
        const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start, const de::Point3D cpy_size,
        const int _memcpy_flag);


    _DECX_API_ de::DH MemcpyLinear(de::Vector& __host, de::GPU_Vector& __device, const int _memcpy_flag);
    _DECX_API_ de::DH MemcpyLinear_Async(de::Vector& __host, de::GPU_Vector& __device, const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH MemcpyLinear(de::Matrix& __host, de::GPU_Matrix& __device, const int _memcpy_flag);
    _DECX_API_ de::DH MemcpyLinear_Async(de::Matrix& __host, de::GPU_Matrix& __device, const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH MemcpyLinear(de::Tensor& __host, de::GPU_Tensor& __device, const int _memcpy_flag);
    _DECX_API_ de::DH MemcpyLinear_Async(de::Tensor& __host, de::GPU_Tensor& __device, const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH MemcpyLinear(de::MatrixArray& __host, de::GPU_MatrixArray& __device, const int _memcpy_flag);
    _DECX_API_ de::DH MemcpyLinear_Async(de::MatrixArray& __host, de::GPU_MatrixArray& __device, const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH MemcpyLinear(de::TensorArray& __host, de::GPU_TensorArray& __device, const int _memcpy_flag);
    _DECX_API_ de::DH MemcpyLinear_Async(de::TensorArray& __host, de::GPU_TensorArray& __device, const int _memcpy_flag, de::DecxStream& S);
}




#endif