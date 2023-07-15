/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _FILL_H_
#define _FILL_H_


#include "../classes/Matrix.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/Vector.h"
#include "../classes/GPU_Vector.h"
#include "../classes/class_utils.h"



namespace de {
    namespace cpu {
        _DECX_API_ de::DH Constant_fp32(de::Vector& src, const float value);

        _DECX_API_ de::DH Constant_int32(de::Vector& src, const int value);

        _DECX_API_ de::DH Constant_fp64(de::Vector& src, const double value);
    }
}



namespace de {
    namespace cpu {
        _DECX_API_ de::DH Constant_fp32(de::Matrix& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::Matrix& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::Matrix& src, const double value);
    }
}


namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Constant_fp32(GPU_Vector& src, const float value);


        _DECX_API_ de::DH Constant_int32(GPU_Vector& src, const int value);


        _DECX_API_ de::DH Constant_fp64(GPU_Vector& src, const double value);
    }
}


namespace de {
    namespace cuda {
        _DECX_API_ de::DH Constant_fp32(de::GPU_Matrix& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::GPU_Matrix& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::GPU_Matrix& src, const double value);
    }
}



namespace de
{
    enum extend_label {
        _EXTEND_NONE_ = 0,
        _EXTEND_REFLECT_ = 1,
        _EXTEND_CONSTANT_ = 2,
    };
}



namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Extend(de::Vector& src, de::Vector& dst, const uint32_t left, const uint32_t right,
            const int border_type, void* val);



        _DECX_API_ de::DH Extend(de::Matrix& src, de::Matrix& dst, const uint32_t left, const uint32_t right,
            const uint32_t top, const uint32_t bottom, const int border_type, void* val);
    }
}





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


    _DECX_API_ de::DH Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
        const int _memcpy_flag);


    _DECX_API_ de::DH Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start, const de::Point3D cpy_size,
        const int _memcpy_flag);


    _DECX_API_ de::DH MemcpyLinear(de::Vector& __host, de::GPU_Vector& __device, const int _memcpy_flag);


    _DECX_API_ de::DH MemcpyLinear(de::Matrix& __host, de::GPU_Matrix& __device, const int _memcpy_flag);


    _DECX_API_ de::DH MemcpyLinear(de::Tensor& __host, de::GPU_Tensor& __device, const int _memcpy_flag);


    _DECX_API_ de::DH MemcpyLinear(de::MatrixArray& __host, de::GPU_MatrixArray& __device, const int _memcpy_flag);


    _DECX_API_ de::DH MemcpyLinear(de::TensorArray& __host, de::GPU_TensorArray& __device, const int _memcpy_flag);
}


#endif