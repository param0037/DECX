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


#include <Algorithms/reduce/CUDA/reduce_callers.cuh>
#include <Classes/Vector.h>
#include <Classes/GPU_Vector.h>
#include <configs/config.h>
#include <cudaStream_management/cudaEvent_queue.h>
#include <Classes/classes_util.h>
#include <Classes/Number.h>


namespace decx
{
    namespace reduce
    {
        static void vector_reduce_sum_fp32(decx::_Vector* src, de::Number* res);


        static void vector_reduce_sum_u8_i32(decx::_Vector* src, de::Number* res);


        static void vector_reduce_sum_fp16(decx::_Vector* src, de::Number* res, const uint32_t _fp16_accu);


        static void dev_vector_reduce_sum_fp32(decx::_GPU_Vector* src, de::Number* res);


        static void dev_vector_reduce_sum_u8_i32(decx::_GPU_Vector* src, de::Number* res);


        static void dev_vector_reduce_sum_fp16(decx::_GPU_Vector* src, de::Number* res, const uint32_t _fp16_accu);
    }
}


static void decx::reduce::vector_reduce_sum_fp32(decx::_Vector* src, de::Number* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(src->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::vector_reduce_sum_u8_i32(decx::_Vector* src, de::Number* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.generate_configs(src->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(uint8_t), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_INT32_);

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::vector_reduce_sum_fp16(decx::_Vector* src, de::Number* res, const uint32_t _fp16_accu)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.set_fp16_accuracy(_fp16_accu);
    _kp_configs.generate_configs(src->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_sum_fp16_caller_Async(&_kp_configs, S, _fp16_accu);

    const uint8_t _cpy_size = (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) ? sizeof(float) : sizeof(de::Half);
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * _cpy_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    res->set_type_flag((_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) ? 
                       de::_DATA_TYPES_FLAGS_::_FP32_ : 
                       de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::dev_vector_reduce_sum_fp32(decx::_GPU_Vector* src, de::Number* res)
{
    uint64_t proc_len_v4 = src->_Length() / 4;
    uint64_t proc_len_v1 = src->Len();

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::dev_vector_reduce_sum_u8_i32(decx::_GPU_Vector* src, de::Number* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_INT32_);

    E->event_record(S);
    E->synchronize();
}


static void decx::reduce::dev_vector_reduce_sum_fp16(decx::_GPU_Vector* src, de::Number* res, const uint32_t _fp16_accu)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.set_fp16_accuracy(_fp16_accu);
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    decx::reduce::cuda_reduce1D_sum_fp16_caller_Async(&_kp_configs, S, _fp16_accu);

    const uint8_t _cpy_size = (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) ? sizeof(float) : sizeof(de::Half);
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * _cpy_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag((_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) ? 
                       de::_DATA_TYPES_FLAGS_::_FP32_ : 
                       de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();
}



#endif