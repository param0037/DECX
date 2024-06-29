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


#ifndef _DATA_TRANSMISSION_CUH_
#define _DATA_TRANSMISSION_CUH_


#include "../../classes/All_classes.h"


namespace de
{
    enum DECX_Memcpy_Flags
    {
        DECX_MEMCPY_H2D = 0,
        DECX_MEMCPY_D2H = 1,
        DECX_MEMCPY_D2D = 1
    };
}


namespace decx
{
    namespace bp 
    {
        template <bool _async_call>
        void Memcpy_Vec(decx::_Vector* __host, decx::_GPU_Vector* __decxvice, const uint64_t start_src, const uint64_t start_dst, const uint64_t cpy_size, 
            const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);

        template <bool _async_call>
        void Memcpy_Mat(decx::_Matrix* __host, decx::_GPU_Matrix* __decxvice, const de::Point2D start_src, const de::Point2D start_dst, const de::Point2D cpy_size,
            const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);

        template <bool _async_call>
        void Memcpy_Tens(decx::_Tensor* __host, decx::_GPU_Tensor* __decxvice, const de::Point3D start_src, const de::Point3D start_dst,
            const de::Point3D cpy_size, const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);

        template <bool _async_call>
        void Memcpy_TensArr(decx::_TensorArray* __host, decx::_GPU_TensorArray* __decxvice, const de::Point3D start_src, const de::Point3D start_dst,
            const de::Point3D cpy_size, const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0);


        template <bool _print>
        void MemcpyLinear_caller(void* __host, void* __device, const uint64_t _host_len, const uint64_t _device_len,
            const int _memcpy_flag, de::DH* handle, const uint32_t stream_id = 0);
    }
}


namespace de
{
    _DECX_API_ de::DH Memcpy(de::Vector& __host, de::GPU_Vector& __device, const uint64_t start_src, const uint64_t start_dst,
        const uint64_t cpy_size, const int _memcpy_flag);


    _DECX_API_ de::DH Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start_src, de::Point2D start_dst, const de::Point2D cpy_size,
        const int _memcpy_flag);


    _DECX_API_ de::DH Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start_src, const de::Point3D start_dst,
        const de::Point3D cpy_size, const int _memcpy_flag);

}


#endif