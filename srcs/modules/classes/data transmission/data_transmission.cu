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


#include "data_transmission.cuh"



template <bool _async_call> void _CRSR_
decx::bp::Memcpy_Vec(decx::_Vector* _host_vec,      decx::_GPU_Vector* _device_vec, 
                     const uint64_t start_src,      const uint64_t start_dst, 
                     const uint64_t cpy_len,        const int _memcpy_flag, 
                     de::DH* handle,                const uint32_t _stream_id)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (_host_vec->Type() != _device_vec->Type()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return;
    }

    const uint8_t& _sizeof = _host_vec->_single_element_size;

    if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_H2D) 
    {
        if (start_src + cpy_len > _host_vec->Len() || start_dst + cpy_len > _device_vec->Len()) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED,
                MEMCPY_OVERRANGED);
            return;
        }

        checkCudaErrors(cudaMemcpyAsync((uint8_t*)_device_vec->Vec.ptr + start_src * _sizeof,
            (uint8_t*)_host_vec->Vec.ptr + start_dst * _sizeof,
            cpy_len * _sizeof, cudaMemcpyHostToDevice, S->get_raw_stream_ref()));
    }
    else {
        if (start_dst + cpy_len > _host_vec->Len() || start_src + cpy_len > _device_vec->Len()) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED,
                MEMCPY_OVERRANGED);
            return;
        }

        checkCudaErrors(cudaMemcpyAsync((uint8_t*)_host_vec->Vec.ptr + start_src * _sizeof,
            (uint8_t*)_device_vec->Vec.ptr + start_dst * _sizeof,
            cpy_len * _sizeof, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _async_call> void _CRSR_
decx::bp::Memcpy_Mat(decx::_Matrix* _host_mat,          decx::_GPU_Matrix* _device_mat, 
                     const de::Point2D start_src,       const de::Point2D start_dst, 
                     const de::Point2D cpy_size,        const int _memcpy_flag, 
                     de::DH* handle,                    const uint32_t _stream_id)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (_host_mat->Type() != _device_mat->Type()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
            TYPE_ERROR_NOT_MATCH);
        return;
    }

    const uint8_t& _sizeof = _host_mat->get_layout()._single_element_size;

    if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_H2D) 
    {
        if (start_src.x + cpy_size.x > _host_mat->Width() || start_dst.x + cpy_size.x > _device_mat->Width() ||
            start_src.y + cpy_size.y > _host_mat->Height() || start_dst.y + cpy_size.y > _device_mat->Height()) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED,
                MEMCPY_OVERRANGED);
            return;
        }

        checkCudaErrors(cudaMemcpy2DAsync(
            DECX_PTR_SHF_XY<void, uint8_t>(_device_mat->Mat.ptr, make_uint2(start_dst.y, start_dst.x * _sizeof), _device_mat->Pitch() * _sizeof),
            _device_mat->Pitch() * _sizeof,
            DECX_PTR_SHF_XY<void, uint8_t>(_host_mat->Mat.ptr, make_uint2(start_src.y, start_src.x * _sizeof), _host_mat->Pitch() * _sizeof),
            _host_mat->Pitch() * _sizeof,
            cpy_size.x * _sizeof,
            cpy_size.y,
            cudaMemcpyHostToDevice, S->get_raw_stream_ref()));
    }
    else {
        if (start_src.x + cpy_size.x > _device_mat->Width() || start_dst.x + cpy_size.x > _host_mat->Width() ||
            start_src.y + cpy_size.y > _device_mat->Height() || start_dst.y + cpy_size.y > _host_mat->Height()) {
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED,
                MEMCPY_OVERRANGED);
            return;
        }

        checkCudaErrors(cudaMemcpy2DAsync(
            DECX_PTR_SHF_XY<void, uint8_t>(_host_mat->Mat.ptr, make_uint2(start_src.y, start_src.x * _sizeof), _host_mat->Pitch() * _sizeof),
            _host_mat->Pitch() * _sizeof,
            DECX_PTR_SHF_XY<void, uint8_t>(_device_mat->Mat.ptr, make_uint2(start_dst.y, start_dst.x * _sizeof), _device_mat->Pitch() * _sizeof),
            _device_mat->Pitch() * _sizeof,
            cpy_size.x * _sizeof,
            cpy_size.y,
            cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _async_call> void _CRSR_
decx::bp::Memcpy_Tens(decx::_Tensor* _host_tensor,              decx::_GPU_Tensor* _device_tensor, 
                      const de::Point3D start_src,              const de::Point3D start_dst, 
                      const de::Point3D cpy_size,               const int _memcpy_flag, 
                      de::DH* handle,                           const uint32_t _stream_id)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return;
    }

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    cudaMemcpy3DParms params = { 0 };
    
    const uint8_t& _sizeof = _host_tensor->get_layout()._single_element_size;
    int3 src_dims, dst_dims;

    params.extent = make_cudaExtent(cpy_size.x * _sizeof, cpy_size.y, cpy_size.z);
    params.srcPos = make_cudaPos(start_src.x * _sizeof, start_src.y, start_src.z);
    params.dstPos = make_cudaPos(start_dst.x * _sizeof, start_dst.y, start_dst.z);

    if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_H2D) 
    {
        params.srcPtr = make_cudaPitchedPtr(_host_tensor->Tens.ptr, _host_tensor->get_layout().dpitch * _sizeof,
            _host_tensor->Depth() * _sizeof, _host_tensor->get_layout().wpitch);

        params.dstPtr = make_cudaPitchedPtr(_device_tensor->Tens.ptr, _device_tensor->get_layout().dpitch * _sizeof,
            _device_tensor->Depth() * _sizeof, _device_tensor->get_layout().wpitch);

        src_dims.x = _host_tensor->Depth();
        src_dims.y = _host_tensor->Width();
        src_dims.z = _host_tensor->Height();

        dst_dims.x = _device_tensor->Depth();
        dst_dims.y = _device_tensor->Width();
        dst_dims.z = _device_tensor->Height();

        params.kind = cudaMemcpyHostToDevice;
    }
    else {
        params.dstPtr = make_cudaPitchedPtr(_host_tensor->Tens.ptr, _host_tensor->get_layout().dpitch * _sizeof,
            _host_tensor->Depth() * _sizeof, _host_tensor->get_layout().wpitch);

        params.srcPtr = make_cudaPitchedPtr(_device_tensor->Tens.ptr, _device_tensor->get_layout().dpitch * _sizeof,
            _device_tensor->Depth() * _sizeof, _device_tensor->get_layout().wpitch);

        src_dims.x = _device_tensor->Depth();
        src_dims.y = _device_tensor->Width();
        src_dims.z = _device_tensor->Height();

        dst_dims.x = _host_tensor->Depth();
        dst_dims.y = _host_tensor->Width();
        dst_dims.z = _host_tensor->Height();

        params.kind = cudaMemcpyDeviceToHost;
    }

    if (start_src.x + cpy_size.x > src_dims.x ||
        start_src.y + cpy_size.y > src_dims.y ||
        start_src.z + cpy_size.z > src_dims.z) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED,
            MEMCPY_OVERRANGED);
        return;
    }
    if (start_dst.x + cpy_size.x > dst_dims.x ||
        start_dst.y + cpy_size.y > dst_dims.y ||
        start_dst.z + cpy_size.z > dst_dims.z) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED,
            MEMCPY_OVERRANGED);
        return;
    }

    checkCudaErrors(cudaMemcpy3DAsync(&params, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}




_DECX_API_ de::DH
de::Memcpy(de::Vector& __host, de::GPU_Vector& __device, const uint64_t start_src, const uint64_t start_dst, const uint64_t cpy_len,
    const int _memcpy_flag)
{
    de::DH handle;

    decx::_Vector* _host_vec = dynamic_cast<decx::_Vector*>(&__host);
    decx::_GPU_Vector* _device_vec = dynamic_cast<decx::_GPU_Vector*>(&__device);

    decx::bp::Memcpy_Vec<false>(_host_vec, _device_vec, start_src, start_dst, cpy_len, _memcpy_flag, &handle);

    return handle;
}



_DECX_API_ de::DH
de::Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start_src, const de::Point2D start_dst, const de::Point2D cpy_size,
    const int _memcpy_flag)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init,
            CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _host_mat = dynamic_cast<decx::_Matrix*>(&__host);
    decx::_GPU_Matrix* _device_mat = dynamic_cast<decx::_GPU_Matrix*>(&__device);

    decx::bp::Memcpy_Mat<false>(_host_mat, _device_mat, start_src, start_dst, cpy_size, _memcpy_flag, &handle);

    return handle;
}


_DECX_API_ de::DH
de::Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start_src, const de::Point3D start_dst, 
    const de::Point3D cpy_size, const int _memcpy_flag)
{
    de::DH handle;

    decx::_Tensor* _host_tensor = dynamic_cast<decx::_Tensor*>(&__host);
    decx::_GPU_Tensor* _device_tensor = dynamic_cast<decx::_GPU_Tensor*>(&__device);

    decx::bp::Memcpy_Tens<false>(_host_tensor, _device_tensor, start_src, start_dst, cpy_size, _memcpy_flag, &handle);

    return handle;
}


template void decx::bp::Memcpy_Vec<true>(decx::_Vector*, decx::_GPU_Vector*, const uint64_t, const uint64_t, const uint64_t, const int, de::DH*, const uint32_t);
template void decx::bp::Memcpy_Vec<false>(decx::_Vector*, decx::_GPU_Vector*, const uint64_t, const uint64_t, const uint64_t, const int, de::DH*, const uint32_t);

template void decx::bp::Memcpy_Mat<true>(decx::_Matrix*, decx::_GPU_Matrix*, const de::Point2D, const de::Point2D, const de::Point2D, const int, de::DH*, const uint32_t);
template void decx::bp::Memcpy_Mat<false>(decx::_Matrix*, decx::_GPU_Matrix*, const de::Point2D,const de::Point2D, const de::Point2D, const int, de::DH*, const uint32_t);

template void decx::bp::Memcpy_Tens<true>(decx::_Tensor*, decx::_GPU_Tensor*, const de::Point3D, const de::Point3D, const de::Point3D, const int, de::DH*, const uint32_t);
template void decx::bp::Memcpy_Tens<false>(decx::_Tensor*, decx::_GPU_Tensor*, const de::Point3D, const de::Point3D, const de::Point3D, const int, de::DH*, const uint32_t);
