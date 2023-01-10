/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderon
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CUDA_FMS_GPU_MATRIX_
#define _CUDA_FMS_GPU_MATRIX_


#include "../../../classes/GPU_Matrix.h"
#include "../Fms_kernel.cuh"
#include "../../../core/basic.h"

using decx::_Matrix;
using decx::_GPU_Matrix;

namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_  de::DH Fms(de::GPU_Matrix<T>& A, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& C, de::GPU_Matrix<T>& dst);



        template <typename T>
        _DECX_API_  de::DH Fms(de::GPU_Matrix<T>& src, T __x, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& dst);
    }
}




template <typename T>
de::DH de::cuda::Fms(de::GPU_Matrix<T>& A, de::GPU_Matrix<T>& B, de::GPU_Matrix<T>& C, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>& _A = dynamic_cast<_GPU_Matrix<T>&>(A);
    _GPU_Matrix<T>& _B = dynamic_cast<_GPU_Matrix<T>&>(B);
    _GPU_Matrix<T>& _C = dynamic_cast<_GPU_Matrix<T>&>(C);
    _GPU_Matrix<T>& _dst = dynamic_cast<_GPU_Matrix<T>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _B.width || _A.height != _B.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A.pitch * (size_t)_A.height;
    decx::dev_Kfms_m(_A.Mat.ptr, _B.Mat.ptr, _C.Mat.ptr, _dst.Mat.ptr, len);

    decx::Success(&handle);
    return handle;
}

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& C, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<int>& A, de::GPU_Matrix<int>& B, de::GPU_Matrix<int>& C, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<de::Half>& A, de::GPU_Matrix<de::Half>& B, de::GPU_Matrix<de::Half>& C, de::GPU_Matrix<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<double>& A, de::GPU_Matrix<double>& B, de::GPU_Matrix<double>& C, de::GPU_Matrix<double>& dst);



template <typename T>
de::DH de::cuda::Fms(de::GPU_Matrix<T>& A, T __x, de::GPU_Matrix<T> &B, de::GPU_Matrix<T>& dst)
{
    _GPU_Matrix<T>& _A = dynamic_cast<_GPU_Matrix<T>&>(A);
    _GPU_Matrix<T>& _B = dynamic_cast<_GPU_Matrix<T>&>(B);
    _GPU_Matrix<T>& _dst = dynamic_cast<_GPU_Matrix<T>&>(dst);

    de::DH handle;
    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    if (_A.width != _dst.width || _A.height != _dst.height) {
        decx::MDim_Not_Matching(&handle);
        return handle;
    }

    const size_t len = (size_t)_A.pitch * (size_t)_A.height;
    decx::dev_Kfms_c(_A.Mat.ptr, __x, _B.Mat.ptr, _dst.Mat.ptr, len);

    decx::Success(&handle);
    return handle;
}

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<float>& A, float __x, de::GPU_Matrix<float>& _B, de::GPU_Matrix<float>& dst);

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<int>& A, int __x, de::GPU_Matrix<int>& _B, de::GPU_Matrix<int>& dst);

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<de::Half>& A, de::Half __x, de::GPU_Matrix<de::Half>& _B, de::GPU_Matrix<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Fms(de::GPU_Matrix<double>& A, double __x, de::GPU_Matrix<double>& _B, de::GPU_Matrix<double>& dst);


#endif