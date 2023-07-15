/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DATA_TRANSMISSION_CUH_
#define _DATA_TRANSMISSION_CUH_


#include "DMA_callers.cuh"
#include "../../classes/All_classes.h"


namespace de
{
    enum DECX_Memcpy_Flags
    {
        DECX_MEMCPY_H2D = 0,
        DECX_MEMCPY_D2H = 1,
    };
}


namespace decx
{
    namespace bp 
    {
        template <bool _print>
        _DECX_API_ void Memcpy_Raw_API(decx::_Vector* __host, decx::_GPU_Vector* __decxvice, const size_t start, const size_t cpy_size, 
            const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void Memcpy_Raw_API(decx::_Matrix* __host, decx::_GPU_Matrix* __decxvice, const de::Point2D start, const de::Point2D cpy_size,
            const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void Memcpy_Raw_API(decx::_Tensor* __host, decx::_GPU_Tensor* __decxvice, const de::Point3D start, const de::Point3D cpy_size,
            const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void MemcpyLinear_Raw_API(decx::_Vector* __host, decx::_GPU_Vector* __device, const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void MemcpyLinear_Raw_API(decx::_Matrix* __host, decx::_GPU_Matrix* __device, const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void MemcpyLinear_Raw_API(decx::_Tensor* __host, decx::_GPU_Tensor* __device, const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void MemcpyLinear_Raw_API(decx::_MatrixArray* __host, decx::_GPU_MatrixArray* __decxvice, const int _memcpy_flag, de::DH* handle);

        template <bool _print>
        _DECX_API_ void MemcpyLinear_Raw_API(decx::_TensorArray* __host, decx::_GPU_TensorArray* __decxvice, const int _memcpy_flag, de::DH* handle);
    }
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