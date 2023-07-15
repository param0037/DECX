/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DOT_FP32_H_
#define _DOT_FP32_H_

#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_Vector.h"
#include "../../../../classes/GPU_Tensor.h"
#include "dot_kernel_fp32.cuh"
#include "../../../basic_process/type_statistics/CUDA/reduction_sum.cuh"

#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"



namespace decx
{
    namespace dot {
        static void Kdot_fp32(float4* A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
            const size_t dev_len, const size_t dst_len, decx::cuda_stream* S);
    }
}



static void decx::dot::Kdot_fp32(float4* A, float4* B, decx::alloc::MIF<float4>* dev_A, decx::alloc::MIF<float4>* dev_B,
    const size_t dev_len, const size_t dst_len, decx::cuda_stream* S)
{
    size_t grid = dev_len / 2, thr_num = dev_len / 2;
    int count = 0;
    while (1) {
        grid = decx::utils::ceil<size_t>(thr_num, REDUCTION_BLOCK_SIZE);

        if (count == 0) {
            decx::dot::GPUK::cu_dot_vec4f_start << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (A, B, dev_A->mem, thr_num, dst_len);
            decx::utils::set_mutex_memory_state<float4, float4>(dev_A, dev_B);
        }
        else {
            if (dev_A->leading) {
                decx::dot::GPUK::cu_dot_vec4f << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (dev_A->mem, dev_B->mem, thr_num, dst_len);
                decx::utils::set_mutex_memory_state<float4, float4>(dev_B, dev_A);
            }
            else {
                decx::dot::GPUK::cu_dot_vec4f << <grid, REDUCTION_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (dev_B->mem, dev_A->mem, thr_num, dst_len);
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
    namespace cuda
    {
        _DECX_API_ de::DH Dot_fp32(de::GPU_Vector& A, de::GPU_Vector& B, float* res);


        _DECX_API_ de::DH Dot_fp32(de::GPU_Matrix& A, de::GPU_Matrix& B, float* res);


        _DECX_API_ de::DH Dot_fp32(de::GPU_Tensor& A, de::GPU_Tensor& B, float* res);
    }
}


#endif