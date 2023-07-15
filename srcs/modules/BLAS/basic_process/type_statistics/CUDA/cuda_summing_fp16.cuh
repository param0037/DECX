/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CUDA_SUMMING_FP16_CUH_
#define _CUDA_SUMMING_FP16_CUH_


#include "../../classes/GPU_Matrix.h"
#include "../../classes/GPU_Vector.h"
#include "../../classes/GPU_Tensor.h"
#include "reduction_sum.cuh"

using decx::_GPU_Matrix;
using decx::_GPU_Vector;
using decx::_GPU_Tensor;


namespace decx
{
    static void _sum_vec4_fp16_1D(de::Half* src, de::Half* res, const size_t dev_len, de::DH* handle);


    static void _sum_vec4_fp16_1D_accu_L1(de::Half* src, de::Half* res, const size_t dev_len, de::DH* handle);


    static void _sum_vec4_fp16_1D_accu_L2(de::Half* src, de::Half* res, const size_t dev_len, de::DH* handle);
}


static void decx::_sum_vec4_fp16_1D(de::Half* src, de::Half* res, const size_t dev_len, de::DH* handle)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    const size_t tmp_dev_size = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size, true, S)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size, true, S)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;
    
    int __iter = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);
        if (__iter == 0) {
            cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
        }
        else {
            if (dev_A.leading) {
                cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A.mem, dev_B.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_B, &dev_A);
            }
            else {
                cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B.mem, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
            }
        }
        thr_num = decx::utils::ceil<size_t>(grid, 2);
        ++__iter;
        if (grid == 1)    break;
    }

    half2_8 ans;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(&ans, dev_A.mem, sizeof(half2_8), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(&ans, dev_B.mem, sizeof(half2_8), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    float _internal_res = 0;
    decx::_final_vec8_sum_fp16(&ans, &_internal_res);
    *(reinterpret_cast<__half*>(res)) = __float2half(_internal_res);

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);

    S->detach();
}



static void decx::_sum_vec4_fp16_1D_accu_L1(de::Half* src, de::Half* res, const size_t dev_len, de::DH* handle)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    const size_t tmp_dev_size = decx::utils::ceil<size_t>(thr_num, 512) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size, true, S)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size, true, S)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    int __iter = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, 512);
        if (__iter == 0) {
            cu_sum_vec8_fp16_accu_fp16_output << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
        }
        else {
            if (dev_A.leading) {
                cu_sum_vec8_fp16_accu_fp16_output << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A.mem, dev_B.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_B, &dev_A);
            }
            else {
                cu_sum_vec8_fp16_accu_fp16_output << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B.mem, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
            }
        }
        thr_num = decx::utils::ceil<size_t>(grid, 2);
        ++__iter;
        if (grid == 1)    break;
    }

    half2_8 ans;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(&ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(&ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    float _internal_res = 0;
    decx::_final_vec8_sum_fp16(&ans, &_internal_res);
    *(reinterpret_cast<__half*>(res)) = __float2half(_internal_res);

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);

    S->detach();
}



static void decx::_sum_vec4_fp16_1D_accu_L2(de::Half* src, de::Half* res, const size_t dev_len, de::DH* handle)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    const size_t tmp_dev_size = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size, true, S)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size, true, S)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    int __iter = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);
        if (__iter == 0) {
            cu_sum_vec8_fp16_accu_fp32_output << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
        }
        else {
            if (dev_A.leading) {
                cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A.mem, dev_B.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_B, &dev_A);
            }
            else {
                cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B.mem, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
            }
        }
        thr_num = decx::utils::ceil<size_t>(grid, 2);
        ++__iter;
        if (grid == 1)    break;
    }

    float4 ans;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(&ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(&ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    float _internal_res = 0;
    decx::_final_vec4_sum<float, float4>(&ans, &_internal_res);
    *(reinterpret_cast<__half*>(res)) = __float2half(_internal_res);

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);

    S->detach();
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Sum(de::GPU_Vector<de::Half>& src, de::Half* res, const int accu_flag);


        _DECX_API_ de::DH Sum(de::GPU_Matrix<de::Half>& src, de::Half* res, const int accu_flag);
    }
}


namespace decx
{
    enum Sum_Fp16_Accuracy_Levels
    {
        Sum_Fp16_Accurate_L0 = 0,
        Sum_Fp16_Accurate_L1 = 1,
        Sum_Fp16_Accurate_L2 = 2,
    };
}


de::DH de::cuda::Sum(de::GPU_Vector<de::Half>& src, de::Half* res, const int accu_flag)
{
    decx::_GPU_Vector<de::Half>* _src = dynamic_cast<decx::_GPU_Vector<de::Half>*>(&src);

    de::DH handle;

    switch (accu_flag)
    {
    case decx::Sum_Fp16_Accuracy_Levels::Sum_Fp16_Accurate_L0:
        decx::_sum_vec4_fp16_1D(_src->Vec.ptr, res, _src->_length / 8, &handle);
        break;

    case decx::Sum_Fp16_Accuracy_Levels::Sum_Fp16_Accurate_L1:
        decx::_sum_vec4_fp16_1D_accu_L1(_src->Vec.ptr, res, _src->_length / 8, &handle);
        break;

    case decx::Sum_Fp16_Accuracy_Levels::Sum_Fp16_Accurate_L2:
        decx::_sum_vec4_fp16_1D_accu_L2(_src->Vec.ptr, res, _src->_length / 8, &handle);
        break;
    }

    decx::err::Success(&handle);
    return handle;
}



de::DH de::cuda::Sum(de::GPU_Matrix<de::Half>& src, de::Half* res, const int accu_flag)
{
    decx::_GPU_Matrix<de::Half>* _src = dynamic_cast<decx::_GPU_Matrix<de::Half>*>(&src);

    de::DH handle;
    switch (accu_flag)
    {
    case decx::Sum_Fp16_Accuracy_Levels::Sum_Fp16_Accurate_L0:
        decx::_sum_vec4_fp16_1D(_src->Mat.ptr, res, _src->_element_num / 8, &handle);
        break;

    case decx::Sum_Fp16_Accuracy_Levels::Sum_Fp16_Accurate_L1:
        decx::_sum_vec4_fp16_1D_accu_L1(_src->Mat.ptr, res, _src->_element_num / 8, &handle);
        break;

    case decx::Sum_Fp16_Accuracy_Levels::Sum_Fp16_Accurate_L2:
        decx::_sum_vec4_fp16_1D_accu_L2(_src->Mat.ptr, res, _src->_element_num / 8, &handle);
        break;
    default:
        break;
    }
    
    decx::err::Success(&handle);
    return handle;
}



#endif