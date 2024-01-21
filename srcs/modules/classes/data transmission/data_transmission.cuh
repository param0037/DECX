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
#include "../../../Async Engine/Async_task_threadpool/Async_Engine.h"
#include "../../../Async Engine/DecxStream/DecxStream.h"


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
        template <bool _async_call>
        void Memcpy_Vec(decx::_Vector* __host, decx::_GPU_Vector* __decxvice, const size_t start, const size_t cpy_size, 
            const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);

        template <bool _async_call>
        void Memcpy_Mat(decx::_Matrix* __host, decx::_GPU_Matrix* __decxvice, const de::Point2D start, const de::Point2D cpy_size,
            const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);

        template <bool _async_call>
        void Memcpy_Tens(decx::_Tensor* __host, decx::_GPU_Tensor* __decxvice, const de::Point3D start, const de::Point3D cpy_size,
            const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);


        template <bool _print>
        void MemcpyLinear_caller(void* __host, void* __device, const uint64_t _host_len, const uint64_t _device_len,
            const int _memcpy_flag, de::DH* handle, const uint32_t stream_id = 0);
    }
}


namespace de
{
    _DECX_API_ de::DH Memcpy(de::Vector& __host, de::GPU_Vector& __device, const size_t start, const size_t cpy_size,
        const int _memcpy_flag);
    _DECX_API_ de::DH Memcpy_Async(de::Vector& __host, de::GPU_Vector& __device, const size_t start, const size_t cpy_size,
        const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
        const int _memcpy_flag);
    _DECX_API_ de::DH Memcpy_Async(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
        const int _memcpy_flag, de::DecxStream& S);


    _DECX_API_ de::DH Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start, const de::Point3D cpy_size,
        const int _memcpy_flag);
    _DECX_API_ de::DH Memcpy_Async(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start, const de::Point3D cpy_size,
        const int _memcpy_flag, de::DecxStream& S);


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