/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _TRANSPOSE_H_
#define _TRANSPOSE_H_


#include "../../../core/basic.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/classes_util.h"
#include "transpose.cuh"
#include "../../../core/cudaStream_management/cudaStream_queue.h"


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Transpose(GPU_Matrix<float>& src, GPU_Matrix<float>& dst);


        _DECX_API_ de::DH Transpose(GPU_Matrix<int>& src, GPU_Matrix<int>& dst);


        _DECX_API_ de::DH Transpose(GPU_Matrix<double>& src, GPU_Matrix<double>& dst);


        _DECX_API_ de::DH Transpose(GPU_Matrix<de::Half>& src, GPU_Matrix<de::Half>& dst);


        _DECX_API_ de::DH Transpose(GPU_Matrix<de::CPf>& src, GPU_Matrix<de::CPf>& dst);
    }
}

using decx::_GPU_Matrix;

namespace decx
{
    /**
    * ATTENTION: sizeof(_vec_type) / sizeof(_type) must be 4
    * @param <_type> : This is the typename of the single element
    * @param <_vec_type> : This is the typename of the vector combined with 4 elements
    */
    template <typename _type, typename _vec_type>
    void dev_transpose_4x4(_GPU_Matrix<_type>* src, _GPU_Matrix<_type>* dst, de::DH* handle);


    /**
    * ATTENTION: sizeof(_vec_type) / sizeof(_type) must be 8
    * @param <_type> : This is the typename of the single element
    * @param <_vec_type> : This is the typename of the vector combined with 8 elements
    */
    void dev_transpose_h8x8(_GPU_Matrix<de::Half>* src, _GPU_Matrix<de::Half>* dst, de::DH* handle);


    /**
    * ATTENTION: sizeof(_vec_type) / sizeof(_type) must be 2
    * @param <_type> : This is the typename of the single element
    * @param <_vec_type> : This is the typename of the vector combined with 2 elements
    */
    template <typename _type, typename _vec_type>
    void dev_transpose_2x2(_GPU_Matrix<_type>* src, _GPU_Matrix<_type>* dst, de::DH* handle);
}



template <typename _type, typename _vec_type>
void decx::dev_transpose_4x4(_GPU_Matrix<_type>* src, _GPU_Matrix<_type>* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);

    dst->re_construct(src->height, src->width);

    _vec_type* dev_src = reinterpret_cast<_vec_type*>(src->Mat.ptr),
             * dev_dst = reinterpret_cast<_vec_type*>(dst->Mat.ptr);
    
    const dim3 grid(decx::utils::ceil<int>(dst->pitch / 4, 16),
        decx::utils::ceil<int>(src->pitch / 4, 16));
    const dim3 block(16, 16);

    cu_transpose_vec4x4<<<grid, block, 0, S->get_raw_stream_ref() >>>(dev_src, dev_dst, src->pitch / 4, dst->pitch / 4, make_uint2(src->width, src->height));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();
}



void decx::dev_transpose_h8x8(_GPU_Matrix<de::Half>* src, _GPU_Matrix<de::Half>* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    
    dst->re_construct(src->height, src->width);

    float4* dev_src = reinterpret_cast<float4*>(src->Mat.ptr),
         * dev_dst = reinterpret_cast<float4*>(dst->Mat.ptr);

    const dim3 grid(decx::utils::ceil<int>(dst->pitch / 8, 16),
        decx::utils::ceil<int>(src->pitch / 8, 16));
    const dim3 block(16, 16);

    cu_transpose_vec8x8 << <grid, block, 0, S->get_raw_stream_ref() >> > (dev_src, dev_dst, src->pitch / 8, dst->pitch / 8, make_uint2(src->width, src->height));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();
}


template <typename _type, typename _vec_type>
void decx::dev_transpose_2x2(_GPU_Matrix<_type>* src, _GPU_Matrix<_type>* dst, de::DH* handle)
{
    decx::cuda_stream* S;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);

    dst->re_construct(src->height, src->width);

    _vec_type* dev_src = reinterpret_cast<_vec_type*>(src->Mat.ptr),
        * dev_dst = reinterpret_cast<_vec_type*>(dst->Mat.ptr);

    const dim3 grid(decx::utils::ceil<int>(dst->pitch / 2, 16),
        decx::utils::ceil<int>(src->pitch / 2, 16));
    const dim3 block(16, 16);

    cu_transpose_vec2x2 << <grid, block, 0, S->get_raw_stream_ref() >> > (dev_src, dev_dst, src->pitch / 2, dst->pitch / 2, make_uint2(src->width, src->height));

    checkCudaErrors(cudaDeviceSynchronize());
    S->detach();
}




de::DH de::cuda::Transpose(GPU_Matrix<float>& src, GPU_Matrix<float>& dst)
{
    de::DH handle;

    decx::Success(&handle);
    _GPU_Matrix<float>* _src = dynamic_cast<_GPU_Matrix<float>*>(&src);
    _GPU_Matrix<float>* _dst = dynamic_cast<_GPU_Matrix<float>*>(&dst);

    decx::dev_transpose_4x4<float, float4>(_src, _dst, &handle);

    return handle;
}


de::DH de::cuda::Transpose(GPU_Matrix<int>& src, GPU_Matrix<int>& dst)
{
    de::DH handle;

    decx::Success(&handle);
    _GPU_Matrix<int>* _src = dynamic_cast<_GPU_Matrix<int>*>(&src);
    _GPU_Matrix<int>* _dst = dynamic_cast<_GPU_Matrix<int>*>(&dst);

    decx::dev_transpose_4x4<int, float4>(_src, _dst, &handle);

    return handle;
}


de::DH de::cuda::Transpose(GPU_Matrix<double>& src, GPU_Matrix<double>& dst)
{
    de::DH handle;

    decx::Success(&handle);
    _GPU_Matrix<double>* _src = dynamic_cast<_GPU_Matrix<double>*>(&src);
    _GPU_Matrix<double>* _dst = dynamic_cast<_GPU_Matrix<double>*>(&dst);

    decx::dev_transpose_2x2<double, double2>(_src, _dst, &handle);

    return handle;
}


de::DH de::cuda::Transpose(GPU_Matrix<de::CPf>& src, GPU_Matrix<de::CPf>& dst)
{
    de::DH handle;

    decx::Success(&handle);
    _GPU_Matrix<de::CPf>* _src = dynamic_cast<_GPU_Matrix<de::CPf>*>(&src);
    _GPU_Matrix<de::CPf>* _dst = dynamic_cast<_GPU_Matrix<de::CPf>*>(&dst);

    decx::dev_transpose_2x2<de::CPf, double2>(_src, _dst, &handle);

    return handle;
}



de::DH de::cuda::Transpose(GPU_Matrix<de::Half>& src, GPU_Matrix<de::Half>& dst)
{
    de::DH handle;

    decx::Success(&handle);
    _GPU_Matrix<de::Half>* _src = dynamic_cast<_GPU_Matrix<de::Half>*>(&src);
    _GPU_Matrix<de::Half>* _dst = dynamic_cast<_GPU_Matrix<de::Half>*>(&dst);

    decx::dev_transpose_h8x8(_src, _dst, &handle);

    return handle;
}


#endif