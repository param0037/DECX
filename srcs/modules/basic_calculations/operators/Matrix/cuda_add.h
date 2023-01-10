/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CUDA_ADD_GPU_MATRIX_H_
#define _CUDA_ADD_GPU_MATRIX_H_


#include "../../../classes/GPU_Matrix.h"
#include "../Add_kernel.cuh"
#include "../../../core/basic.h"

using decx::_Matrix;
using decx::_GPU_Matrix;


namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_  de::DH Add(de::GPU_Matrix<T>& A, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& dst);


        template <typename T>
        _DECX_API_  de::DH Add(de::GPU_Matrix<T>& src, T __x, de::GPU_Matrix<T>& dst);
    }
}




template <typename T>
de::DH de::cuda::Add(de::GPU_Matrix<T>& A, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>& _A = dynamic_cast<_GPU_Matrix<T>&>(A);
    _GPU_Matrix<T>& _B = dynamic_cast<_GPU_Matrix<T>&>(B);
    _GPU_Matrix<T>& _dst = dynamic_cast<_GPU_Matrix<T>&>(dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.width || _A.height != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A.pitch * (size_t)_A.height;
    decx::dev_Kadd_m(_A.Mat.ptr, _B.Mat.ptr, _dst.Mat.ptr, len);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<int>& A, de::GPU_Matrix<int>& B, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<double>& A, de::GPU_Matrix<double>& B, de::GPU_Matrix<double>& dst);



template <typename T>
de::DH de::cuda::Add(de::GPU_Matrix<T>& src, T __x, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>& _src = dynamic_cast<_GPU_Matrix<T>&>(src);
    _GPU_Matrix<T>& _dst = dynamic_cast<_GPU_Matrix<T>&>(dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_src.width != _dst.width || _src.height != _dst.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_src.pitch * (size_t)_src.height;
    decx::dev_Kadd_c(_src.Mat.ptr, __x, _dst.Mat.ptr, len);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<float>& src, float __x, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<int>& src, int __x, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<de::Half>& src, de::Half __x, de::GPU_Matrix<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Matrix<double>& src, double __x, de::GPU_Matrix<double>& dst);


#endif          // #ifndef _CUDA_ADD_GPU_MATRIX_H_