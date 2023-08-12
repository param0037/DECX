/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _VECTOR_REDUCE_SUM_CUH_
#define _VECTOR_REDUCE_SUM_CUH_

#include "../../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Vector.h"
#include "../../../../core/configs/config.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../classes/classes_util.h"


namespace decx
{
    namespace reduce
    {
        static void vector_reduce_sum_fp32(decx::_Vector* src, float* res);


        static void vector_reduce_sum_u8_i32(decx::_Vector* src, int32_t* res);


        static void vector_reduce_sum_fp16_fp32(decx::_Vector* src, float* res);


        static void dev_vector_reduce_sum_fp32(decx::_GPU_Vector* src, float* res);


        static void dev_vector_reduce_sum_u8_i32(decx::_GPU_Vector* src, int32_t* res);


        static void dev_vector_reduce_sum_fp16_fp32(decx::_GPU_Vector* src, float* res);
    }
}


static void decx::reduce::vector_reduce_sum_fp32(decx::_Vector* src, float* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(src->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::vector_reduce_sum_u8_i32(decx::_Vector* src, int32_t* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.generate_configs(src->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(uint8_t), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_dst(), 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::vector_reduce_sum_fp16_fp32(decx::_Vector* src, float* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.generate_configs(src->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_sum_fp16_fp32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::dev_vector_reduce_sum_fp32(decx::_GPU_Vector* src, float* res)
{
    uint64_t proc_len_v4 = src->_Length() / 4;
    uint64_t proc_len_v1 = src->Len();

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::dev_vector_reduce_sum_u8_i32(decx::_GPU_Vector* src, int32_t* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_dst(), 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::dev_vector_reduce_sum_fp16_fp32(decx::_GPU_Vector* src, float* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    decx::reduce::cuda_reduce1D_sum_fp16_fp32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



#endif