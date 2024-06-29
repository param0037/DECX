/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the ZLIB License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* This software is provided 'as-is', without any express or implied warranty. In no event 
* will the authors be held liable for any damages arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose, including 
* commercial applications, and to alter it and redistribute it freely, subject to the 
* following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not claim that you 
*    wrote the original software. If you use this software in a product, an acknowledgment 
*    in the product documentation would be appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be misrepresented 
*    as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/


#ifndef _CUDA_ADD_GPU_TENSOR_H_
#define _CUDA_ADD_GPU_TENSOR_H_

#include "../../../classes/GPU_Tensor.h"
#include "../Add_kernel.cuh"
#include "../../../core/basic.h"


using decx::_GPU_Tensor;

namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_  de::DH Add(de::GPU_Tensor<T>& A, de::GPU_Tensor<T>& B, de::GPU_Tensor<T>& dst);


        template <typename T>
        _DECX_API_  de::DH Add(de::GPU_Tensor<T>& src, T __x, de::GPU_Tensor<T>& dst);
    }
}


template <typename T>
de::DH de::cuda::Add(de::GPU_Tensor<T>& A, de::GPU_Tensor<T>& B, de::GPU_Tensor<T>& dst)
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

    decx::dev_Kadd_m_2D(_A.Tens.ptr, _B.Tens.ptr, _dst.Tens.ptr, eq_pitch, bounds);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<float>& A, de::GPU_Tensor<float>& B, de::GPU_Tensor<float>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<int>& A, de::GPU_Tensor<int>& B, de::GPU_Tensor<int>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<de::Half>& A, de::GPU_Tensor<de::Half>& B, de::GPU_Tensor<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<double>& A, de::GPU_Tensor<double>& B, de::GPU_Tensor<double>& dst);



template <typename T>
de::DH de::cuda::Add(de::GPU_Tensor<T>& src, T __x, de::GPU_Tensor<T>& dst)
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

    decx::dev_Kadd_c_2D(_src.Tens.ptr, __x, _dst.Tens.ptr, eq_pitch, bounds);

    return handle;
}

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<float>& src, float __x, de::GPU_Tensor<float>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<int>& src, int __x, de::GPU_Tensor<int>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<de::Half>& src, de::Half __x, de::GPU_Tensor<de::Half>& dst);

template _DECX_API_ de::DH de::cuda::Add(de::GPU_Tensor<double>& src, double __x, de::GPU_Tensor<double>& dst);




#endif