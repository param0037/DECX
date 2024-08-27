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
    _DECX_API_ de::DH Memcpy(de::Vector& __host, de::GPU_Vector& __device, const uint64_t start_src, const uint64_t start_dst,
        const uint64_t cpy_size, const int _memcpy_flag);
    
    
    // [W, H]
    _DECX_API_ de::DH Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start_src, de::Point2D start_dst, const de::Point2D cpy_size,
        const int _memcpy_flag);
    
    
    // [D, W, H]
    _DECX_API_ de::DH Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start_src, const de::Point3D start_dst,
        const de::Point3D cpy_size, const int _memcpy_flag);
}




#endif