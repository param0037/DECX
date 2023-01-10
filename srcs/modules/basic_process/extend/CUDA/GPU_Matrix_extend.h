/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _GPU_MATRIX_EXTend_H_
#define _GPU_MATRIX_EXTend_H_

#include "../../../classes/GPU_Matrix.h"
#include "sym_ext.cuh"

namespace de
{
    namespace cuda
    {
        template <typename T>
        _DECX_API_ de::DH Extend(de::GPU_Matrix<T>& src, de::GPU_Matrix<T>& dst,
            const uint top, const uint bottom, const uint left, const uint right);
    }
}


namespace decx
{

}


using decx::_GPU_Matrix;

template <typename T>
de::DH de::cuda::Extend(de::GPU_Matrix<T>& src, de::GPU_Matrix<T>& dst,
    const uint top, const uint bottom, const uint left, const uint right)
{
    de::DH handle;

    _GPU_Matrix<T>* _src = dynamic_cast<_GPU_Matrix<T>*>(&src);
    _GPU_Matrix<T>* _dst = dynamic_cast<_GPU_Matrix<T>*>(&dst);

    const uint2 dst_dim = make_uint2(_src->width + left + right,
        _src->height + top + bottom);

    _dst->re_construct(dst_dim.x, dst_dim.y);

    cudaStream_t S;
    cudaStreamCreate(&S);

    T* in_ptr = _dst->Mat.ptr + (size_t)top * _dst->pitch + (size_t)left;

    checkCudaErrors(cudaMemcpy2DAsync(in_ptr, _dst->pitch * sizeof(T),
        _src->Mat.ptr, _src->pitch * sizeof(T), _src->width * sizeof(T), _src->height,
        cudaMemcpyDeviceToDevice, S));

    // judge where to start
    float4* start_ptr_BT = reinterpret_cast<float4*>(_dst->Mat.ptr) + (left / (sizeof(float4) / sizeof(T)));

    // top
    const dim3 block_T(8, 32);
    const dim3 grid_T(decx::utils::ceil<uint>(top, 8), decx::utils::ceil<uint>(_dst->pitch / 4, 32));
    cu_sym_ext_T_vec4 << <grid_T, block_T, 0, S >> > (start_ptr_BT, make_uint2(_dst->pitch / 4, top), _dst->pitch / 4);

    // bottom
    const dim3 grid_B(decx::utils::ceil<uint>(bottom, 8), decx::utils::ceil<uint>(_dst->pitch / 4, 32));
    cu_sym_ext_B_vec4 << <grid_B, block_T, 0, S >> > (
        start_ptr_BT, make_uint2(_dst->pitch / 4, bottom), _dst->pitch / 4, top + _src->height);

    // left
    const dim3 block_L(32, 8);
    const dim3 grid_L(decx::utils::ceil<uint>(_dst->height, 32), decx::utils::ceil<uint>(left, 8));
    cu_sym_ext_L << <grid_L, block_L, 0, S >> > (_dst->Mat.ptr, make_uint2(left, _dst->height), _dst->pitch);

    // right
    const dim3 grid_R(decx::utils::ceil<uint>(_dst->height, 32), decx::utils::ceil<uint>(right, 8));
    cu_sym_ext_R << <grid_R, block_L, 0, S >> > (
        _dst->Mat.ptr, make_uint2(right, _dst->height), _dst->pitch, left + _src->width);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    return handle;
}

template _DECX_API_ de::DH de::cuda::Extend(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst,
    const uint top, const uint bottom, const uint left, const uint right);

#endif