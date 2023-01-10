/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _DOT_FP16_H_
#define _DOT_FP16_H_

#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../classes/GPU_Tensor.h"
#include "dot_kernel_fp16.cuh"


namespace decx
{
    enum Dot_Fp16_Accuracy_Levels
    {
        Dot_Fp16_Accurate_L0 = 0,
        Dot_Fp16_Accurate_L1 = 1,
        Dot_Fp16_Accurate_L2 = 2,
    };
}



namespace decx
{
    static void Kdot_fp16(float4 *A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B, 
        const size_t dev_len, const size_t dst_len, decx::cuda_stream* S);


    static void Kdot_fp16_accu_L1(float4 *A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B, 
        const size_t dev_len, const size_t dst_len, decx::cuda_stream* S);


    static void Kdot_fp16_accu_L2(float4* A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
        const size_t dev_len, const size_t dst_len, decx::cuda_stream* S);
}



static void decx::Kdot_fp16(float4* A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
    const size_t dev_len, const size_t dst_len, decx::cuda_stream* S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);

        if (count == 0) {
            decx::dot::GPUK::cu_dot_vec8_fp16_start << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                A, B, dev_A->mem, thr_num, dst_len);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
        }
        else {
            if (dev_A->leading) {
                decx::bp::GPUK::cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A->mem, dev_B->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                decx::bp::GPUK::cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B->mem, dev_A->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
            }
        }
        thr_num = decx::utils::ceil<size_t>(grid, 2);
        ++count;

        if (grid == 1)    break;
    }
}



void decx::Kdot_fp16_accu_L1(float4* A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
    const size_t dev_len, const size_t dst_len, decx::cuda_stream* S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);

        if (count == 0) {
            decx::dot::GPUK::cu_dot_vec8h_start_accu_fp16_output << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                A, B, dev_A->mem, thr_num, dst_len);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
        }
        else {
            if (dev_A->leading) {
                decx::bp::GPUK::cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A->mem, dev_B->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                decx::bp::GPUK::cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B->mem, dev_A->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
            }
        }
        thr_num = decx::utils::ceil<size_t>(grid, 2);
        ++count;

        if (grid == 1)    break;
    }
}




void decx::Kdot_fp16_accu_L2(float4* A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
    const size_t dev_len, const size_t dst_len, decx::cuda_stream* S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);

        if (count == 0) {
            decx::dot::GPUK::cu_dot_vec8h_start_accu_fp32_output << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                A, B, dev_A->mem, thr_num, dst_len);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
        }
        else {
            if (dev_A->leading) {
                decx::bp::GPUK::cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A->mem, dev_B->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                decx::bp::GPUK::cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B->mem, dev_A->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
            }
        }
        thr_num = decx::utils::ceil<size_t>(grid, 2);
        ++count;

        if (grid == 1)    break;
    }
}



namespace decx
{
    namespace cuda {
        void Dot_fp16_vec(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::Half* res, de::DH* handle, const int accu_flag);


        void Dot_fp16_mat(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, de::Half* res, de::DH* handle, const int accu_flag);


        void Dot_fp16_ten(decx::_GPU_Tensor* A, decx::_GPU_Tensor* B, de::Half* res, de::DH* handle, const int accu_flag);
    }
}



void decx::cuda::Dot_fp16_vec(
    decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::Half* res, de::DH* handle, const int accu_flag)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = A->_length / 8;
    const size_t tmp_dev_size = 
        decx::utils::ceil<size_t>(dev_len / 2, REDUCTION_BLOCK_SIZE);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    float4* ans = new float4();
    half2_8* ans_half2_8_ptr = NULL;

    switch (accu_flag)
    {
    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L0:
        decx::Kdot_fp16((float4*)A->Vec.ptr, (float4*)B->Vec.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);
        break;

    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L1:
        decx::Kdot_fp16_accu_L1((float4*)A->Vec.ptr, (float4*)B->Vec.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);
        break;

    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L2:
        decx::Kdot_fp16_accu_L2((float4*)A->Vec.ptr, (float4*)B->Vec.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size * 2, S);
        break;

    default:
        break;
    }

    float _ans = 0;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    if (accu_flag == 2) {
        _ans += ans->x;             _ans += ans->y;
        _ans += ans->z;             _ans += ans->w;
    }
    else {
        half2_8* ans_half2_8_ptr = reinterpret_cast<half2_8*>(ans);

        _ans += __half2float(ans_half2_8_ptr->x.x);         _ans += __half2float(ans_half2_8_ptr->x.y);
        _ans += __half2float(ans_half2_8_ptr->y.x);         _ans += __half2float(ans_half2_8_ptr->y.y);
        _ans += __half2float(ans_half2_8_ptr->z.x);         _ans += __half2float(ans_half2_8_ptr->z.y);
        _ans += __half2float(ans_half2_8_ptr->w.x);         _ans += __half2float(ans_half2_8_ptr->w.y);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    *(reinterpret_cast<__half*>(res)) = __float2half(_ans);
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}



void decx::cuda::Dot_fp16_mat(
    decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, de::Half* res, de::DH* handle, const int accu_flag)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = A->_element_num / 8;
    const size_t tmp_dev_size =
        decx::utils::ceil<size_t>(dev_len / 2, REDUCTION_BLOCK_SIZE);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    float4* ans = new float4();
    half2_8* ans_half2_8_ptr = NULL;

    switch (accu_flag)
    {
    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L0:
        decx::Kdot_fp16((float4*)A->Mat.ptr, (float4*)B->Mat.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);
        break;

    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L1:
        decx::Kdot_fp16_accu_L1((float4*)A->Mat.ptr, (float4*)B->Mat.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);
        break;

    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L2:
        decx::Kdot_fp16_accu_L2((float4*)A->Mat.ptr, (float4*)B->Mat.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size * 2, S);
        break;

    default:
        break;
    }

    float _ans = 0;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    if (accu_flag == 2) {
        _ans += ans->x;             _ans += ans->y;
        _ans += ans->z;             _ans += ans->w;
    }
    else {
        half2_8* ans_half2_8_ptr = reinterpret_cast<half2_8*>(ans);

        _ans += __half2float(ans_half2_8_ptr->x.x);         _ans += __half2float(ans_half2_8_ptr->x.y);
        _ans += __half2float(ans_half2_8_ptr->y.x);         _ans += __half2float(ans_half2_8_ptr->y.y);
        _ans += __half2float(ans_half2_8_ptr->z.x);         _ans += __half2float(ans_half2_8_ptr->z.y);
        _ans += __half2float(ans_half2_8_ptr->w.x);         _ans += __half2float(ans_half2_8_ptr->w.y);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    *(reinterpret_cast<__half*>(res)) = __float2half(_ans);
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}




void decx::cuda::Dot_fp16_ten(
    decx::_GPU_Tensor* A, decx::_GPU_Tensor* B, de::Half* res, de::DH* handle, const int accu_flag)
{
    // 2x float4 as a pair of adding operation
    const size_t dev_len = A->_element_num / 8;
    const size_t tmp_dev_size =
        decx::utils::ceil<size_t>(dev_len / 2, REDUCTION_BLOCK_SIZE);

    decx::PtrInfo<float4> dev_tmp1, dev_tmp2;
    if (decx::alloc::_device_malloc(&dev_tmp1, tmp_dev_size * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&dev_tmp2, tmp_dev_size * sizeof(float4))) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::alloc::MIF<float4> dev_A, dev_B;
    dev_A.mem = dev_tmp1.ptr;
    dev_B.mem = dev_tmp2.ptr;

    float4* ans = new float4();
    half2_8* ans_half2_8_ptr = NULL;

    switch (accu_flag)
    {
    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L0:
        decx::Kdot_fp16((float4*)A->Tens.ptr, (float4*)B->Tens.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);
        break;

    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L1:
        decx::Kdot_fp16_accu_L1((float4*)A->Tens.ptr, (float4*)B->Tens.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size, S);
        break;

    case decx::Dot_Fp16_Accuracy_Levels::Dot_Fp16_Accurate_L2:
        decx::Kdot_fp16_accu_L2((float4*)A->Tens.ptr, (float4*)B->Tens.ptr, &dev_A, &dev_B, dev_len, tmp_dev_size * 2, S);
        break;

    default:
        break;
    }

    float _ans = 0;
    if (dev_A.leading) {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_A.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }
    else {
        checkCudaErrors(cudaMemcpyAsync(
            ans, dev_B.mem, sizeof(float4), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    }

    if (accu_flag == 2) {
        _ans += ans->x;             _ans += ans->y;
        _ans += ans->z;             _ans += ans->w;
    }
    else {
        half2_8* ans_half2_8_ptr = reinterpret_cast<half2_8*>(ans);

        _ans += __half2float(ans_half2_8_ptr->x.x);         _ans += __half2float(ans_half2_8_ptr->x.y);
        _ans += __half2float(ans_half2_8_ptr->y.x);         _ans += __half2float(ans_half2_8_ptr->y.y);
        _ans += __half2float(ans_half2_8_ptr->z.x);         _ans += __half2float(ans_half2_8_ptr->z.y);
        _ans += __half2float(ans_half2_8_ptr->w.x);         _ans += __half2float(ans_half2_8_ptr->w.y);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    *(reinterpret_cast<__half*>(res)) = __float2half(_ans);
    delete ans;

    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);
    S->detach();
}




namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Dot_fp16(de::GPU_Vector& A, de::GPU_Vector& B, de::Half* res, const int accu_flag);


        _DECX_API_ de::DH Dot_fp16(de::GPU_Matrix& A, de::GPU_Matrix& B, de::Half* res, const int accu_flag);


        _DECX_API_ de::DH Dot_fp16(de::GPU_Tensor& A, de::GPU_Tensor& B, de::Half* res, const int accu_flag);
    }
}



#endif