/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CUDA_DIV_GPU_TENSOR_H_
#define _CUDA_DIV_GPU_TENSOR_H_

#include "../../../classes/GPU_Tensor.h"
#include "../Div_kernel.cuh"
#include "../../../core/basic.h"


using decx::_GPU_Tensor;

namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_  de::DH Div(de::GPU_Tensor<T>& A, de::GPU_Tensor<T>& B, de::GPU_Tensor<T>& dst);


        template <typename T>
        _DECX_API_  de::DH Div(de::GPU_Tensor<T>& src, T __x, de::GPU_Tensor<T>& dst);


        template <typename T>
        _DECX_API_  de::DH Div(T __x, de::GPU_Tensor<T>& src, de::GPU_Tensor<T>& dst);
    }
}


template <typename T>
de::DH de::cuda::Div(de::GPU_Tensor<T>& A, de::GPU_Tensor<T>& B, de::GPU_Tensor<T>& dst)
{
    _GPU_Tensor<T>& _A = dynamic_cast<_GPU_Tensor<T>&>(A);
    _GPU_Tensor<T>& _B = dynamic_cast<_GPU_Tensor<T>&>(B);
    _GPU_Tensor<T>& _dst = dynamic_cast<_GPU_Tensor<T>&>(dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }

    _dst.re_construct(_A.width, _A.height, _A.depth);
    
    const size_t eq_pitch = _A.dp_x_wp;
    const uint2 bounds = make_uint2(_A.width * _A.dpitch, _A.height);

    decx::dev_Kdiv_m_2D(_A.Tens.ptr, _B.Tens.ptr, _dst.Tens.ptr, eq_pitch, bounds);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<float>& A, de::GPU_Tensor<float>& B, de::GPU_Tensor<float>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<int>& A, de::GPU_Tensor<int>& B, de::GPU_Tensor<int>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<de::Half>& A, de::GPU_Tensor<de::Half>& B, de::GPU_Tensor<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<double>& A, de::GPU_Tensor<double>& B, de::GPU_Tensor<double>& dst);



template <typename T>
de::DH de::cuda::Div(de::GPU_Tensor<T>& src, T __x, de::GPU_Tensor<T>& dst)
{
    _GPU_Tensor<T>& _src = dynamic_cast<_GPU_Tensor<T>&>(src);
    _GPU_Tensor<T>& _dst = dynamic_cast<_GPU_Tensor<T>&>(dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }
    _dst.re_construct(_src.width, _src.height, _src.depth);

    const size_t eq_pitch = _src.dp_x_wp;
    const uint2 bounds = make_uint2(_src.width * _src.dpitch, _src.height);

    decx::dev_Kdiv_c_2D(_src.Tens.ptr, __x, _dst.Tens.ptr, eq_pitch, bounds);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<float>& src, float __x, de::GPU_Tensor<float>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<int>& src, int __x, de::GPU_Tensor<int>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<de::Half>& src, de::Half __x, de::GPU_Tensor<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::GPU_Tensor<double>& src, double __x, de::GPU_Tensor<double>& dst);



template <typename T>
de::DH de::cuda::Div(T __x, de::GPU_Tensor<T>& src, de::GPU_Tensor<T>& dst)
{
    _GPU_Tensor<T>& _src = dynamic_cast<_GPU_Tensor<T>&>(src);
    _GPU_Tensor<T>& _dst = dynamic_cast<_GPU_Tensor<T>&>(dst);

    de::DH handle;
    decx::Success(&handle);

    if (!decx::cuP.is_init) {
        decx::Not_init(&handle);
        return handle;
    }
    _dst.re_construct(_src.width, _src.height, _src.depth);

    const size_t eq_pitch = _src.dp_x_wp;
    const uint2 bounds = make_uint2(_src.width * _src.dpitch, _src.height);

    decx::dev_Kdiv_cinv_2D(_src.Tens.ptr, __x, _dst.Tens.ptr, eq_pitch, bounds);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Div(float __x, de::GPU_Tensor<float>& src, de::GPU_Tensor<float>& dst);

template _DECX_API_ de::DH de::cuda::Div(int __x, de::GPU_Tensor<int>& src, de::GPU_Tensor<int>& dst);

template _DECX_API_ de::DH de::cuda::Div(de::Half __x, de::GPU_Tensor<de::Half>& src, de::GPU_Tensor<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Div(double __x, de::GPU_Tensor<double>& src, de::GPU_Tensor<double>& dst);



#endif