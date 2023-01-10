/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _CUDA_REVERSE_CUH_
#define _CUDA_REVERSE_CUH_


#include "../../../core/basic.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/classes_util.h"


namespace de
{
    namespace cuda
    {
        template<typename T>
        de::DH Reverse(de::GPU_Matrix<T>& src, de::GPU_Matrix<T>& dst);
    }
}




using decx::_GPU_Matrix;

template<typename T>
de::DH de::cuda::Reverse(de::GPU_Matrix<T>& src, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>* _src = dynamic_cast<_GPU_Matrix<T>*>(&src);
    _GPU_Matrix<T>* _dst = dynamic_cast<_GPU_Matrix<T>*>(&dst);

    de::DH handle;
    decx::Success(&handle);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));
    //decx::cuda_stream* S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);

    const uint2 original_dim = make_uint2(_src->width, _src->height);
    decx::PtrInfo<T> buffer;

    if (decx::alloc::_device_malloc(&buffer, original_dim.x * original_dim.y * sizeof(T))) {
        Print_Error_Message(4, ALLOC_FAIL);
        decx::err::AllocateFailure(&handle);
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy2DAsync(buffer.ptr, _src->width * sizeof(T),
        _src->Mat.ptr, _src->pitch * sizeof(T), _src->width * sizeof(T), _src->height,
        cudaMemcpyDeviceToDevice, S));

    thrust::device_ptr<T> th_buffer = thrust::device_pointer_cast<T>(buffer.ptr);

    thrust::reverse(thrust::device, th_buffer, th_buffer + original_dim.x * original_dim.y);

    checkCudaErrors(cudaMemcpy2DAsync(_dst->Mat.ptr, _dst->pitch * sizeof(T),
        buffer.ptr, _src->width * sizeof(T), _src->width * sizeof(T), _src->height,
        cudaMemcpyDeviceToDevice, S));

    //S->detach();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    return handle;
}


template _DECX_API_ de::DH de::cuda::Reverse(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Reverse(de::GPU_Matrix<int>& src, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Reverse(de::GPU_Matrix<double>& src, de::GPU_Matrix<double>& dst);

template _DECX_API_ de::DH de::cuda::Reverse(de::GPU_Matrix<de::Half>& src, de::GPU_Matrix<de::Half>& dst);


#endif