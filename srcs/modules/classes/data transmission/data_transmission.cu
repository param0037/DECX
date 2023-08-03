/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "data_transmission.cuh"


template <bool _async_call> void
decx::bp::Memcpy_Vec(decx::_Vector* _host_vec, decx::_GPU_Vector* _device_vec, const size_t start, const size_t cpy_len,
    const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(handle);
        return;
    }

    if (start + cpy_len > _host_vec->length) {
        decx::err::InvalidParam<true>(handle);
        return;
    }

    if (_host_vec->Type() != _device_vec->Type()) {
        if (_host_vec->_single_element_size != _device_vec->_single_element_size) {
            decx::err::TypeError_NotMatch<true>(handle);
            return;
        }
        else {
            decx::warn::Memcpy_different_types<true>(handle);
        }
    }

    const void* _src_ptr = NULL;

    switch (_memcpy_flag)
    {
    case de::DECX_Memcpy_Flags::DECX_MEMCPY_H2D:
        _src_ptr = (uint8_t*)_host_vec->Vec.ptr + (size_t)start * (size_t)_host_vec->_single_element_size;
        if (_async_call) {
            decx::bp::_DMA_memcpy1D_sync(_src_ptr, _device_vec->Vec.ptr, cpy_len * _host_vec->_single_element_size, cudaMemcpyHostToDevice);
        }
        else {
            decx::async::register_async_task(_stream_id, decx::bp::_DMA_memcpy1D_sync, 
                _src_ptr, _device_vec->Vec.ptr, cpy_len * _host_vec->_single_element_size, cudaMemcpyHostToDevice);
        }
        break;

    case de::DECX_Memcpy_Flags::DECX_MEMCPY_D2H:
        _src_ptr = (uint8_t*)_device_vec->Vec.ptr + (size_t)start * (size_t)_host_vec->_single_element_size;
        if (_async_call) {
            decx::bp::_DMA_memcpy1D_sync(_src_ptr, _host_vec->Vec.ptr, cpy_len * _host_vec->_single_element_size, cudaMemcpyDeviceToHost);
        }
        else {
            decx::async::register_async_task(_stream_id, decx::bp::_DMA_memcpy1D_sync,
                _src_ptr, _host_vec->Vec.ptr, cpy_len * _host_vec->_single_element_size, cudaMemcpyDeviceToHost);
        }
        break;

    default:
        decx::err::MeaningLessFlag<true>(handle);
        break;
    }

    if (handle->error_type == decx::DECX_error_types::DECX_SUCCESS) {
        decx::err::Success<true>(handle);
    }
}



template <bool _async_call> void
decx::bp::Memcpy_Mat(decx::_Matrix* _host_mat, decx::_GPU_Matrix* _device_mat, const de::Point2D start, const de::Point2D cpy_size,
    const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(handle);
        return;
    }

    if (start.x + cpy_size.x > _host_mat->Width() || start.y + cpy_size.y > _host_mat->Height()) {
        decx::err::InvalidParam<true>(handle);
        return;
    }

    if (_host_mat->Type() != _device_mat->Type()) {
        if (_host_mat->get_layout()._single_element_size != _device_mat->get_layout()._single_element_size) {
            decx::err::TypeError_NotMatch<true>(handle);
            return;
        }
        else {
            decx::warn::Memcpy_different_types<true>(handle);
        }
    }

    if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_H2D)
    {
        decx::bp::memcpy2D_multi_dims_optimizer _opt(_host_mat->Mat.ptr, _host_mat->get_layout(), _device_mat->Mat.ptr, _device_mat->get_layout());
        
        _opt.memcpy2D_optimizer(make_uint2(_host_mat->Height(), _host_mat->Width()), make_uint2(start.x, start.y), make_uint2(cpy_size.x, cpy_size.y));
        _opt.execute_DMA<_async_call>(cudaMemcpyHostToDevice, handle, _stream_id);
    }
    else if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_D2H)
    {
        decx::bp::memcpy2D_multi_dims_optimizer _opt(_device_mat->Mat.ptr, _device_mat->get_layout(), _host_mat->Mat.ptr, _host_mat->get_layout());
        
        _opt.memcpy2D_optimizer(make_uint2(_device_mat->Height(), _device_mat->Width()), make_uint2(start.x, start.y), make_uint2(cpy_size.x, cpy_size.y));
        _opt.execute_DMA<_async_call>(cudaMemcpyDeviceToHost, handle, _stream_id);
    }
    else {
        decx::err::MeaningLessFlag<true>(handle);
    }

    if (handle->error_type == decx::DECX_error_types::DECX_SUCCESS) {
        decx::err::Success<true>(handle);
    }
}



template <bool _async_call> void
decx::bp::Memcpy_Tens(decx::_Tensor* _host_tensor, decx::_GPU_Tensor* _device_tensor, const de::Point3D start, const de::Point3D cpy_size,
    const int _memcpy_flag, de::DH* handle, const uint32_t _stream_id = 0)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(handle);
        return;
    }

    if (start.x + cpy_size.x > _host_tensor->Width() ||
        start.y + cpy_size.y > _host_tensor->Height() ||
        start.z + cpy_size.z > _host_tensor->Depth()) {
        decx::err::InvalidParam<true>(handle);
        return;
    }

    if (_host_tensor->Type() != _device_tensor->Type()) {
        if (_host_tensor->_layout._single_element_size != _device_tensor->_layout._single_element_size) {
            decx::err::TypeError_NotMatch<true>(handle);
            return;
        }
        else {
            decx::warn::Memcpy_different_types<true>(handle);
        }
    }

    if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_H2D)
    {
        decx::bp::memcpy3D_multi_dims_optimizer _opt(_host_tensor->_layout, _host_tensor->Tens.ptr,
            _device_tensor->_layout, _device_tensor->Tens.ptr);

        _opt.memcpy3D_optimizer(
            make_uint3(_host_tensor->Depth(), _host_tensor->Width(), _host_tensor->Height()),
            make_uint3(start.z, start.x, start.y),
            make_uint3(cpy_size.z, cpy_size.x, cpy_size.y));
        _opt.execute_DMA<_async_call>(cudaMemcpyHostToDevice, handle, _stream_id);
    }
    else if (_memcpy_flag == de::DECX_Memcpy_Flags::DECX_MEMCPY_D2H)
    {
        decx::bp::memcpy3D_multi_dims_optimizer _opt(_device_tensor->_layout, _device_tensor->Tens.ptr,
            _host_tensor->_layout, _host_tensor->Tens.ptr);

        _opt.memcpy3D_optimizer(
            make_uint3(_device_tensor->Depth(), _device_tensor->Width(), _device_tensor->Height()),
            make_uint3(start.z, start.x, start.y),
            make_uint3(cpy_size.z, cpy_size.x, cpy_size.y));
        _opt.execute_DMA<_async_call>(cudaMemcpyDeviceToHost, handle, _stream_id);
    }
    else {
        decx::err::MeaningLessFlag<true>(handle);
    }

    if (handle->error_type == decx::DECX_error_types::DECX_SUCCESS) {
        decx::err::Success<true>(handle);
    }
}



_DECX_API_ de::DH
de::Memcpy(de::Vector& __host, de::GPU_Vector& __device, const size_t start, const size_t cpy_len,
    const int _memcpy_flag)
{
    de::DH handle;

    decx::_Vector* _host_vec = dynamic_cast<decx::_Vector*>(&__host);
    decx::_GPU_Vector* _device_vec = dynamic_cast<decx::_GPU_Vector*>(&__device);

    decx::bp::Memcpy_Vec<false>(_host_vec, _device_vec, start, cpy_len, _memcpy_flag, &handle);

    return handle;
}


_DECX_API_ de::DH
de::Memcpy_Async(de::Vector& __host, de::GPU_Vector& __device, const size_t start, const size_t cpy_len,
    const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_Vector* _host_vec = dynamic_cast<decx::_Vector*>(&__host);
    decx::_GPU_Vector* _device_vec = dynamic_cast<decx::_GPU_Vector*>(&__device);

    decx::bp::Memcpy_Vec<true>(_host_vec, _device_vec, start, cpy_len, _memcpy_flag, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH
de::Memcpy(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
    const int _memcpy_flag)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _host_mat = dynamic_cast<decx::_Matrix*>(&__host);
    decx::_GPU_Matrix* _device_mat = dynamic_cast<decx::_GPU_Matrix*>(&__device);

    decx::bp::Memcpy_Mat<false>(_host_mat, _device_mat, start, cpy_size, _memcpy_flag, &handle);

    return handle;
}


_DECX_API_ de::DH
de::Memcpy_Async(de::Matrix& __host, de::GPU_Matrix& __device, const de::Point2D start, const de::Point2D cpy_size,
    const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _host_mat = dynamic_cast<decx::_Matrix*>(&__host);
    decx::_GPU_Matrix* _device_mat = dynamic_cast<decx::_GPU_Matrix*>(&__device);

    decx::bp::Memcpy_Mat<true>(_host_mat, _device_mat, start, cpy_size, _memcpy_flag, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH
de::Memcpy(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start, const de::Point3D cpy_size,
    const int _memcpy_flag)
{
    de::DH handle;

    decx::_Tensor* _host_tensor = dynamic_cast<decx::_Tensor*>(&__host);
    decx::_GPU_Tensor* _device_tensor = dynamic_cast<decx::_GPU_Tensor*>(&__device);

    decx::bp::Memcpy_Tens<false>(_host_tensor, _device_tensor, start, cpy_size, _memcpy_flag, &handle);

    return handle;
}


_DECX_API_ de::DH
de::Memcpy_Async(de::Tensor& __host, de::GPU_Tensor& __device, const de::Point3D start, const de::Point3D cpy_size,
    const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_Tensor* _host_tensor = dynamic_cast<decx::_Tensor*>(&__host);
    decx::_GPU_Tensor* _device_tensor = dynamic_cast<decx::_GPU_Tensor*>(&__device);

    decx::bp::Memcpy_Tens<true>(_host_tensor, _device_tensor, start, cpy_size, _memcpy_flag, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear(de::Vector& __host, de::GPU_Vector& __device, const int _memcpy_flag)
{
    de::DH handle;

    decx::_Vector* _host_Vec = dynamic_cast<decx::_Vector*>(&__host);
    decx::_GPU_Vector* _device_Vec = dynamic_cast<decx::_GPU_Vector*>(&__device);

    decx::bp::MemcpyLinear_caller<false>(_host_Vec->Vec.ptr, _device_Vec->Vec.ptr,
        _host_Vec->total_bytes, _device_Vec->total_bytes, _memcpy_flag, &handle);

    return handle;
}


_DECX_API_ de::DH
de::MemcpyLinear_Async(de::Vector& __host, de::GPU_Vector& __device, const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_Vector* _host_Vec = dynamic_cast<decx::_Vector*>(&__host);
    decx::_GPU_Vector* _device_Vec = dynamic_cast<decx::_GPU_Vector*>(&__device);

    decx::bp::MemcpyLinear_caller<true>(_host_Vec->Vec.ptr, _device_Vec->Vec.ptr,
        _host_Vec->total_bytes, _device_Vec->total_bytes, _memcpy_flag, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear(de::Matrix& __host, de::GPU_Matrix& __device, const int _memcpy_flag)
{
    de::DH handle;

    decx::_Matrix* _host_Mat = dynamic_cast<decx::_Matrix*>(&__host);
    decx::_GPU_Matrix* _device_Mat = dynamic_cast<decx::_GPU_Matrix*>(&__device);

    decx::bp::MemcpyLinear_caller<false>(_host_Mat->Mat.ptr, _device_Mat->Mat.ptr, 
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle);

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear_Async(de::Matrix& __host, de::GPU_Matrix& __device, const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_Matrix* _host_Mat = dynamic_cast<decx::_Matrix*>(&__host);
    decx::_GPU_Matrix* _device_Mat = dynamic_cast<decx::_GPU_Matrix*>(&__device);

    decx::bp::MemcpyLinear_caller<true>(_host_Mat->Mat.ptr, _device_Mat->Mat.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear(de::Tensor& __host, de::GPU_Tensor& __device, const int _memcpy_flag)
{
    de::DH handle;

    decx::_Tensor* _host_Mat = dynamic_cast<decx::_Tensor*>(&__host);
    decx::_GPU_Tensor* _device_Mat = dynamic_cast<decx::_GPU_Tensor*>(&__device);

    decx::bp::MemcpyLinear_caller<false>(_host_Mat->Tens.ptr, _device_Mat->Tens.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle);

    return handle;
}


_DECX_API_ de::DH
de::MemcpyLinear_Async(de::Tensor& __host, de::GPU_Tensor& __device, const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_Tensor* _host_Mat = dynamic_cast<decx::_Tensor*>(&__host);
    decx::_GPU_Tensor* _device_Mat = dynamic_cast<decx::_GPU_Tensor*>(&__device);

    decx::bp::MemcpyLinear_caller<true>(_host_Mat->Tens.ptr, _device_Mat->Tens.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear(de::MatrixArray& __host, de::GPU_MatrixArray& __device, const int _memcpy_flag)
{
    de::DH handle;

    decx::_MatrixArray* _host_Mat = dynamic_cast<decx::_MatrixArray*>(&__host);
    decx::_GPU_MatrixArray* _device_Mat = dynamic_cast<decx::_GPU_MatrixArray*>(&__device);

    decx::bp::MemcpyLinear_caller<false>(_host_Mat->MatArr.ptr, _device_Mat->MatArr.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle);

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear_Async(de::MatrixArray& __host, de::GPU_MatrixArray& __device, const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_MatrixArray* _host_Mat = dynamic_cast<decx::_MatrixArray*>(&__host);
    decx::_GPU_MatrixArray* _device_Mat = dynamic_cast<decx::_GPU_MatrixArray*>(&__device);

    decx::bp::MemcpyLinear_caller<true>(_host_Mat->MatArr.ptr, _device_Mat->MatArr.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle, S.Get_ID());

    return handle;
}




_DECX_API_ de::DH
de::MemcpyLinear(de::TensorArray& __host, de::GPU_TensorArray& __device, const int _memcpy_flag)
{
    de::DH handle;

    decx::_TensorArray* _host_Mat = dynamic_cast<decx::_TensorArray*>(&__host);
    decx::_GPU_TensorArray* _device_Mat = dynamic_cast<decx::_GPU_TensorArray*>(&__device);

    decx::bp::MemcpyLinear_caller<false>(_host_Mat->TensArr.ptr, _device_Mat->TensArr.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle);

    return handle;
}



_DECX_API_ de::DH
de::MemcpyLinear_Async(de::TensorArray& __host, de::GPU_TensorArray& __device, const int _memcpy_flag, de::DecxStream& S)
{
    de::DH handle;

    decx::_TensorArray* _host_Mat = dynamic_cast<decx::_TensorArray*>(&__host);
    decx::_GPU_TensorArray* _device_Mat = dynamic_cast<decx::_GPU_TensorArray*>(&__device);

    decx::bp::MemcpyLinear_caller<true>(_host_Mat->TensArr.ptr, _device_Mat->TensArr.ptr,
        _host_Mat->get_total_bytes(), _device_Mat->get_total_bytes(), _memcpy_flag, &handle, S.Get_ID());

    return handle;
}


template <bool _is_async> void
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


template void decx::bp::Memcpy_Vec<true>(decx::_Vector*, decx::_GPU_Vector*, const size_t, const size_t, const int, de::DH*, const uint32_t);
template void decx::bp::Memcpy_Vec<false>(decx::_Vector*, decx::_GPU_Vector*, const size_t, const size_t, const int, de::DH*, const uint32_t);

template void decx::bp::Memcpy_Mat<true>(decx::_Matrix*, decx::_GPU_Matrix*, const de::Point2D, const de::Point2D, const int, de::DH*, const uint32_t);
template void decx::bp::Memcpy_Mat<false>(decx::_Matrix*, decx::_GPU_Matrix*, const de::Point2D,const de::Point2D, const int, de::DH*, const uint32_t);

template void decx::bp::Memcpy_Tens<true>(decx::_Tensor*, decx::_GPU_Tensor*, const de::Point3D, const de::Point3D, const int, de::DH*, const uint32_t);
template void decx::bp::Memcpy_Tens<false>(decx::_Tensor*, decx::_GPU_Tensor*, const de::Point3D, const de::Point3D, const int, de::DH*, const uint32_t);


template void decx::bp::MemcpyLinear_caller<true>(void*, void*, const uint64_t, const uint64_t, const int, de::DH*, const uint32_t);
template void decx::bp::MemcpyLinear_caller<false>(void*, void*, const uint64_t, const uint64_t, const int, de::DH*, const uint32_t);