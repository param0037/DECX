/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "../../core/basic.h"
#include "../../classes/Matrix.h"
#include "../../classes/classes_util.h"
#include "max_pooling.cuh"


using decx::_Matrix;
using decx::_GPU_Matrix;

namespace de
{
    namespace nn
    {
        _DECX_API_ de::DH max_pooling_f(de::Matrix<float>& src, de::Matrix<float>& dst, const de::Point2D scale);


        _DECX_API_ de::DH max_pooling_f_recorded(de::Matrix<float>& src, de::Matrix<float>& dst, de::Matrix<int>& index_map, const de::Point2D scale);


        _DECX_API_ de::DH max_pooling_f_recorded(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<int>& index_map, const de::Point2D scale);


        _DECX_API_ de::DH up_sampling(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<int>& index_map, const de::Point2D scale);
    }
}


namespace decx
{
    /*
    * The width of source matrix must be 8x(float4 x2)(device memory)
    */
    void max_pooling_f_caller(decx::_Matrix<float>* src, decx::_Matrix<float>* dst, const de::Point2D scale, de::DH *handle);



    void max_pooling_f_caller_recorded(
        decx::_Matrix<float>* src, decx::_Matrix<float>* dst, decx::_Matrix<int> *index_map, const de::Point2D scale, de::DH* handle);



    void dev_max_pooling_f_caller_recorded(
        decx::_GPU_Matrix<float>* src, decx::_GPU_Matrix<float>* dst, decx::_GPU_Matrix<int>* index_map, const de::Point2D scale, de::DH* handle);


    void dev_up_sampling_f_caller_recorded(
        decx::_GPU_Matrix<float>* src, decx::_GPU_Matrix<float>* dst, decx::_GPU_Matrix<int>* index_map, const de::Point2D scale, de::DH* handle);
}






void decx::max_pooling_f_caller(decx::_Matrix<float>* src, decx::_Matrix<float>* dst, const de::Point2D scale, de::DH* handle)
{
#ifndef GNU_CPUcodes
    /* This algorithm ensures that the input width is two/three times larger than the output one, so do not need to
    * worry about whether the index will out of range while loading from source matrix
    */
    const int2 dev_src_domain = make_int2(decx::utils::ceil<int>(src->width, 4), src->height);
    const int2 dev_dst_domain = make_int2(decx::utils::ceil<int>(dst->width, 4), dst->height);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::PtrInfo<float4> dev_tmp;
    decx::alloc::_device_malloc<float4>(&dev_tmp,
        ((size_t)dev_src_domain.x * (size_t)dev_src_domain.y + (size_t)dev_dst_domain.x * (size_t)dev_dst_domain.y) * sizeof(float4));

    if (dev_tmp.ptr == NULL) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    
    float4* dev_src = dev_tmp.ptr;
    float4* dev_dst = dev_tmp.ptr + (size_t)dev_src_domain.x * (size_t)dev_src_domain.y;

    checkCudaErrors(cudaMemcpy2DAsync(dev_src, dev_src_domain.x * sizeof(float4),
        src->Mat.ptr, src->pitch * sizeof(float), src->width * sizeof(float), src->height,
        cudaMemcpyHostToDevice, S));

    dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);
    dim3 grid(decx::utils::ceil<int>(dev_dst_domain.y, _BLOCK_DEFAULT_), 
              decx::utils::ceil<int>(dev_dst_domain.x, _BLOCK_DEFAULT_));

    switch (scale.y)    // Height_scale
    {
    case 2:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            cu_max_pooling2x2_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;

        case 3:
            cu_max_pooling2x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;
        default:
            break;
        }
        break;
    }
    case 3:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            cu_max_pooling3x2_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;
        case 3:
            cu_max_pooling3x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
    
    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(float),
        dev_dst, dev_dst_domain.x * sizeof(float4), dst->width * sizeof(float), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc<float4>(&dev_tmp);
#else

#endif
}



de::DH de::nn::max_pooling_f(de::Matrix<float>& src, de::Matrix<float>& dst, const de::Point2D scale)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_Matrix<float>* _src = dynamic_cast<decx::_Matrix<float>*>(&src);
    decx::_Matrix<float>* _dst = dynamic_cast<decx::_Matrix<float>*>(&dst);
    if (_src->width % scale.x != 0 || _src->height % scale.y != 0) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    if (_src->width / scale.x != _dst->width || _src->height / scale.y != _dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    decx::max_pooling_f_caller(_src, _dst, scale, &handle);

    return handle;
}



void decx::max_pooling_f_caller_recorded(decx::_Matrix<float>* src, decx::_Matrix<float>* dst, decx::_Matrix<int>* index_map, const de::Point2D scale, de::DH* handle)
{
#ifndef GNU_CPUcodes
    /* This algorithm ensures that the input width is two/three times larger than the output one, so do not need to
    * worry about whether the index will out of range while loading from source matrix
    */
    const int2 dev_src_domain = make_int2(decx::utils::ceil<int>(src->width, 4), src->height);
    const int2 dev_dst_domain = make_int2(decx::utils::ceil<int>(dst->width, 4), dst->height);

    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    decx::PtrInfo<float4> dev_tmp;
    size_t src_size = (size_t)dev_src_domain.x * (size_t)dev_src_domain.y;
    size_t dst_size = (size_t)dev_dst_domain.x * (size_t)dev_dst_domain.y;

    if (decx::alloc::_device_malloc<float4>(&dev_tmp, (src_size + (dst_size << 1)) * sizeof(float4))) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    float4* dev_src = dev_tmp.ptr;
    float4* dev_dst = dev_tmp.ptr + src_size;
    int4* dev_dex_map = reinterpret_cast<int4*>(dev_tmp.ptr) + src_size + dst_size;

    checkCudaErrors(cudaMemcpy2DAsync(dev_src, dev_src_domain.x * sizeof(float4),
        src->Mat.ptr, src->pitch * sizeof(float), src->width * sizeof(float), src->height,
        cudaMemcpyHostToDevice, S));

    dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);
    dim3 grid(decx::utils::ceil<int>(dev_dst_domain.y, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(dev_dst_domain.x, _BLOCK_DEFAULT_));

    switch (scale.y)    // Height_scale
    {
    case 2:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            cu_max_pooling2x2_f_with_dex << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dex_map, dev_dst_domain.x, dev_dst_domain.y);
            break;

        case 3:
            cu_max_pooling2x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;
        default:
            break;
        }
        break;
    }
    case 3:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            cu_max_pooling3x2_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;
        case 3:
            cu_max_pooling3x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr, dst->pitch * sizeof(float),
        dev_dst, dev_dst_domain.x * sizeof(float4), dst->width * sizeof(float), dst->height,
        cudaMemcpyDeviceToHost, S));
    checkCudaErrors(cudaMemcpy2DAsync(index_map->Mat.ptr, dst->pitch * sizeof(int),
        dev_dex_map, dev_dst_domain.x * sizeof(int4), dst->width * sizeof(int), dst->height,
        cudaMemcpyDeviceToHost, S));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));

    decx::alloc::_device_dealloc<float4>(&dev_tmp);
#else

#endif
}




void decx::dev_max_pooling_f_caller_recorded(
    decx::_GPU_Matrix<float>* src, decx::_GPU_Matrix<float>* dst, decx::_GPU_Matrix<int>* index_map, const de::Point2D scale, de::DH* handle)
{
    /* This algorithm ensures that the input width is two/three times larger than the output one, so do not need to
    * worry about whether the index will out of range while loading from source matrix
    */
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);
    dim3 grid(decx::utils::ceil<int>(dst->height, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(dst->pitch / 4, _BLOCK_DEFAULT_));

    switch (scale.y)    // Height_scale
    {
    case 2:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            cu_max_pooling2x2_f_with_dex << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(src->Mat.ptr), 
                reinterpret_cast<float4*>(dst->Mat.ptr), 
                reinterpret_cast<int4*>(index_map->Mat.ptr), 
                dst->pitch / 4, dst->height);
            break;

        case 3:
            /*cu_max_pooling2x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);*/
            break;
        default:
            break;
        }
        break;
    }
    case 3:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            /*cu_max_pooling3x2_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);*/
            break;
        case 3:
            /*cu_max_pooling3x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);*/
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));
}




void decx::dev_up_sampling_f_caller_recorded(
    decx::_GPU_Matrix<float>* src, decx::_GPU_Matrix<float>* dst, decx::_GPU_Matrix<int>* index_map, const de::Point2D scale, de::DH* handle)
{
    /* This algorithm ensures that the input width is two/three times larger than the output one, so do not need to
    * worry about whether the index will out of range while loading from source matrix
    */
    cudaStream_t S;
    checkCudaErrors(cudaStreamCreate(&S));

    dim3 block(_BLOCK_DEFAULT_, _BLOCK_DEFAULT_);
    dim3 grid(decx::utils::ceil<int>(src->height, _BLOCK_DEFAULT_),
        decx::utils::ceil<int>(src->pitch / 4, _BLOCK_DEFAULT_));

    switch (scale.y)    // Height_scale
    {
    case 2:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            cu_up_sampling2x2_f << <grid, block, 0, S >> > (
                reinterpret_cast<float4*>(src->Mat.ptr),
                reinterpret_cast<float4*>(dst->Mat.ptr),
                reinterpret_cast<int4*>(index_map->Mat.ptr),
                src->pitch / 4, src->height);
            break;

        case 3:
            /*cu_max_pooling2x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);*/
            break;
        default:
            break;
        }
        break;
    }
    case 3:
    {
        switch (scale.x)    // Width_scale
        {
        case 2:
            /*cu_max_pooling3x2_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);*/
            break;
        case 3:
            /*cu_max_pooling3x3_f << <grid, block, 0, S >> > (
                dev_src, dev_dst, dev_dst_domain.x, dev_dst_domain.y);*/
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(S));
}





de::DH de::nn::max_pooling_f_recorded(de::Matrix<float>& src, de::Matrix<float>& dst, de::Matrix<int>&index_map, const de::Point2D scale)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_Matrix<float>* _src = dynamic_cast<decx::_Matrix<float>*>(&src);
    decx::_Matrix<float>* _dst = dynamic_cast<decx::_Matrix<float>*>(&dst);
    decx::_Matrix<int>* _index_map = dynamic_cast<decx::_Matrix<int>*>(&index_map);

    if (_src->width % scale.x != 0 || _src->height % scale.y != 0) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    if (_src->width / scale.x != _dst->width || _src->height / scale.y != _dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    decx::max_pooling_f_caller_recorded(_src, _dst, _index_map, scale, &handle);

    return handle;
}



de::DH de::nn::max_pooling_f_recorded(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<int>& index_map, const de::Point2D scale)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_GPU_Matrix<float>* _src = dynamic_cast<decx::_GPU_Matrix<float>*>(&src);
    decx::_GPU_Matrix<float>* _dst = dynamic_cast<decx::_GPU_Matrix<float>*>(&dst);
    decx::_GPU_Matrix<int>* _index_map = dynamic_cast<decx::_GPU_Matrix<int>*>(&index_map);

    if (_src->width % scale.x != 0 || _src->height % scale.y != 0) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    if (_src->width / scale.x != _dst->width || _src->height / scale.y != _dst->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    decx::dev_max_pooling_f_caller_recorded(_src, _dst, _index_map, scale, &handle);

    return handle;
}




de::DH de::nn::up_sampling(de::GPU_Matrix<float>& src, de::GPU_Matrix<float>& dst, de::GPU_Matrix<int>& index_map, const de::Point2D scale)
{
    de::DH handle;
    decx::Success(&handle);

    decx::_GPU_Matrix<float>* _src = dynamic_cast<decx::_GPU_Matrix<float>*>(&src);
    decx::_GPU_Matrix<float>* _dst = dynamic_cast<decx::_GPU_Matrix<float>*>(&dst);
    decx::_GPU_Matrix<int>* _index_map = dynamic_cast<decx::_GPU_Matrix<int>*>(&index_map);

    if (_src->width % scale.x != 0 || _src->height % scale.y != 0) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    if (_dst->width / scale.x != _src->width || _dst->height / scale.y != _src->height) {
        Print_Error_Message(4, MAT_DIM_NOT_MATCH);
        decx::Matrix_number_not_matching(&handle);
        return handle;
    }

    decx::dev_up_sampling_f_caller_recorded(_src, _dst, _index_map, scale, &handle);

    return handle;
}