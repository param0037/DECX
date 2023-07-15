/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CUDA_SUMMING_H_
#define _CUDA_SUMMING_H_


#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_Vector.h"
#include "reduction_sum.cuh"



namespace decx
{
    namespace bp {
        static void _sum_vec4_fp32_1D(float* src, float* res, const size_t dev_len, de::DH* handle);


        static void _sum_fp16(float4* A, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
            const size_t dev_len, const size_t dst_len, decx::cuda_stream* S);
    }
}



static void decx::bp::_sum_fp16(float4* A, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
    const size_t dev_len, const size_t dst_len, decx::cuda_stream* S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);

        if (count == 0) {
            decx::bp::GPUK::cu_sum_vec8_fp16 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                A, dev_A->mem, thr_num, dst_len);
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




static void decx::bp::_sum_vec4_fp32_1D(float* src, float *res, const size_t dev_len, de::DH *handle)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    const size_t tmp_dev_size = decx::utils::ceil<size_t>(thr_num, 512) * sizeof(float4);

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
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
        if (__iter == 0){
            decx::bp::GPUK::cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                (float4*)src, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
            decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
        }
        else {
            if (dev_A.leading) {
                decx::bp::GPUK::cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_A.mem, dev_B.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_B, &dev_A);
            }
            else {
                decx::bp::GPUK::cu_sum_vec4_fp32 << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                    dev_B.mem, dev_A.mem, thr_num, tmp_dev_size / sizeof(float4));
                decx::utils::set_mutex_memory_state<float4, float4>(&dev_A, &dev_B);
            }
        }
        thr_num = grid / 2;
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

    decx::_final_vec4_sum<float, float4>(&ans, res);

    checkCudaErrors(cudaDeviceSynchronize());
    decx::alloc::_device_dealloc(&dev_tmp1);
    decx::alloc::_device_dealloc(&dev_tmp2);

    S->detach();
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Sum_fp32(de::GPU_Vector& src, float* res);

        _DECX_API_ de::DH Sum_fp16(de::GPU_Vector& src, de::Half* res);

        _DECX_API_ de::DH Sum_fp32(de::GPU_Matrix& src, float* res);

        _DECX_API_ de::DH Sum_fp16(de::GPU_Matrix& src, de::Half* res);
    }
}



#endif