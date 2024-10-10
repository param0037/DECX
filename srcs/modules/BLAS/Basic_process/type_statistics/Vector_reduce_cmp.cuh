/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _VECTOR_REDUCE_CMP_CUH_
#define _VECTOR_REDUCE_CMP_CUH_

#include <Algorithms/reduce/CUDA/reduce_callers.cuh>
#include <Classes/Vector.h>
#include <Classes/GPU_Vector.h>
#include <configs/config.h>
#include <cudaStream_management/cudaEvent_queue.h>
#include <Classes/classes_util.h>


namespace decx
{
    namespace reduce
    {
        template <bool _is_max>
        static void vector_reduce_cmp_fp64(decx::_Vector* src, de::Number *res);


        template <bool _is_max>
        static void vector_reduce_cmp_fp32(decx::_Vector* src, de::Number* res);


        template <bool _is_max>
        static void vector_reduce_cmp_fp16(decx::_Vector* src, de::Number* res);


        template <bool _is_max>
        static void vector_reduce_cmp_u8(decx::_Vector* src, de::Number* res);


        template <bool _is_max>
        static void dev_vector_reduce_cmp_fp64(decx::_GPU_Vector* src, de::Number* res);


        template <bool _is_max>
        static void dev_vector_reduce_cmp_fp32(decx::_GPU_Vector* src, de::Number* res);


        template <bool _is_max>
        static void dev_vector_reduce_cmp_fp16(decx::_GPU_Vector* src, de::Number* res);


        template <bool _is_max>
        static void dev_vector_reduce_cmp_u8(decx::_GPU_Vector* src, de::Number* res);
    }
}


template <bool _is_max>
static void decx::reduce::vector_reduce_cmp_fp32(decx::_Vector* src, de::Number* res)
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
    _kp_configs.set_cmp_or_not(true);
    _kp_configs.generate_configs(src->Len(), S);

    _kp_configs.set_fill_val(((float*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<_is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}



template <bool _is_max>
static void decx::reduce::vector_reduce_cmp_fp64(decx::_Vector* src, de::Number* res)
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

    decx::reduce::cuda_reduce1D_configs<double> _kp_configs;
    _kp_configs.set_cmp_or_not(true);
    _kp_configs.generate_configs(src->Len(), S);

    _kp_configs.set_fill_val(((double*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(double), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<_is_max>(&_kp_configs, S);
    
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP64_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::vector_reduce_cmp_fp16(decx::_Vector* src, de::Number* res)
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
    _kp_configs.set_cmp_or_not(true);
    _kp_configs.generate_configs(src->Len(), S);

    _kp_configs.set_fill_val(((de::Half*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<_is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}



template <bool _is_max>
static void decx::reduce::vector_reduce_cmp_u8(decx::_Vector* src, de::Number* res)
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
    _kp_configs.set_cmp_or_not(true);
    _kp_configs.generate_configs(src->Len(), S);

    _kp_configs.set_fill_val(((uint8_t*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_src(), src->Vec.ptr, src->Len() * sizeof(uint8_t), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<_is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_UINT8_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}


// ------------------------------------------- on GPU -------------------------------------------



template <bool _is_max>
static void decx::reduce::dev_vector_reduce_cmp_fp64(decx::_GPU_Vector* src, de::Number* res)
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

    decx::reduce::cuda_reduce1D_configs<double> _kp_configs;
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    double _fill_val = 0;
    checkCudaErrors(cudaMemcpyAsync(src->Vec.ptr, &_fill_val, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    _kp_configs.set_fill_val(_fill_val);

    decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<_is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP64_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::dev_vector_reduce_cmp_fp32(decx::_GPU_Vector* src, de::Number* res)
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
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    float _fill_val = 0;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Vec.ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    _kp_configs.set_fill_val(_fill_val);

    decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<_is_max>(&_kp_configs, S);
    
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}



template <bool _is_max>
static void decx::reduce::dev_vector_reduce_cmp_fp16(decx::_GPU_Vector* src, de::Number* res)
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
    _kp_configs.generate_configs(src->Vec, src->Len(), S);

    de::Half _fill_val;
    _fill_val.val = 0;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Vec.ptr, 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    _kp_configs.set_fill_val(_fill_val);

    decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<_is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}



template <bool _is_max>
static void decx::reduce::dev_vector_reduce_cmp_u8(decx::_GPU_Vector* src, de::Number* res)
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

    uint8_t _fill_val = 0;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Vec.ptr, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    _kp_configs.set_fill_val(_fill_val);

    decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<_is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr<void>(), _kp_configs.get_dst(), 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_UINT8_);

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}


#endif
