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

        template <bool _print>
        _DECX_API_ void Memcpy_Raw_API(decx::_Tensor* __host, decx::_GPU_Tensor* __decxvice, const de::Point3D start, const de::Point3D cpy_size,
            const int _memcpy_flag, de::DH* handle);


        template <bool _print>
        static void MemcpyLinear_caller(void* __host, void* __device, const uint64_t _host_len, const uint64_t _device_len,
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


template <bool _is_async> static void
decx::bp::MemcpyLinear_caller(void* __host, void* __device, const uint64_t _host_size, const uint64_t _device_size,
    const int _memcpy_flag, de::DH* handle, const uint32_t stream_id)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(handle);
        return;
    }

    const void* src_ptr = NULL;
    void* dst_ptr = NULL;
    cudaMemcpyKind translated_cpy_flag;
    uint64_t src_size = 0, dst_size = 0;

    if (_memcpy_flag == de::DECX_MEMCPY_H2D) {
        src_ptr = __host;
        dst_ptr = __device;
        translated_cpy_flag = cudaMemcpyHostToDevice;
        src_size = _host_size;
        dst_size = _device_size;
    }
    else {
        src_ptr = __device;
        dst_ptr = __host;
        translated_cpy_flag = cudaMemcpyDeviceToHost;
        src_size = _device_size;
        dst_size = _host_size;
    }

    if (src_size > dst_size) {
        decx::err::Memcpy_overranged<true>(handle);
        return;
    }

    const uint64_t cpy_size = min(src_size, dst_size);

    if (_is_async) {
        decx::async::register_async_task(stream_id, decx::bp::_DMA_memcpy1D_sync, src_ptr, dst_ptr,
            cpy_size, translated_cpy_flag);
    }
    else {
        decx::bp::_DMA_memcpy1D_sync(src_ptr, dst_ptr, cpy_size, translated_cpy_flag);
    }

    if (handle->error_type == decx::DECX_error_types::DECX_SUCCESS) {
        decx::err::Success<true>(handle);
    }
}


#endif