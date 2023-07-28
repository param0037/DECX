/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


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
        template <bool _print>
        static void vector_reduce_sum_fp32(decx::_Vector* src, float* res, de::DH* handle);


        template <bool _print>
        static void vector_reduce_sum_u8_i32(decx::_Vector* src, int32_t* res, de::DH* handle);


        template <bool _print>
        static void vector_reduce_sum_fp16_fp32(decx::_Vector* src, float* res, de::DH* handle);


        template <bool _print>
        static void dev_vector_reduce_sum_fp32(decx::_GPU_Vector* src, float* res, de::DH* handle);


        template <bool _print>
        static void dev_vector_reduce_sum_u8_i32(decx::_GPU_Vector* src, int32_t* res, de::DH* handle);


        template <bool _print>
        static void dev_vector_reduce_sum_fp16_fp32(decx::_GPU_Vector* src, float* res, de::DH* handle);
    }
}


template <bool _print>
static void decx::reduce::vector_reduce_sum_fp32(decx::_Vector* src, float* res, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs<_print>(src->Len(), S, handle);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_dev_tmp1().ptr, src->Vec.ptr, src->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_fp32_caller_Async<float, false>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


template <bool _print>
static void decx::reduce::vector_reduce_sum_u8_i32(decx::_Vector* src, int32_t* res, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.generate_configs<_print>(src->Len(), S, handle);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_dev_tmp1().ptr, src->Vec.ptr, src->Len() * sizeof(uint8_t), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_u8_i32_caller_Async<uint8_t, false>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


template <bool _print>
static void decx::reduce::vector_reduce_sum_fp16_fp32(decx::_Vector* src, float* res, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.generate_configs<_print>(src->Len(), S, handle);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_dev_tmp1().ptr, src->Vec.ptr, src->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_fp16_fp32_caller_Async<de::Half, false>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


template <bool _print>
static void decx::reduce::dev_vector_reduce_sum_fp32(decx::_GPU_Vector* src, float* res, de::DH* handle)
{
    uint64_t proc_len_v4 = src->_Length() / 4;
    uint64_t proc_len_v1 = src->Len();

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs<_print>(src->Vec, src->Len(), S, handle);

    decx::reduce::cuda_reduce1D_fp32_caller_Async<float, true>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}




template <bool _print>
static void decx::reduce::dev_vector_reduce_sum_u8_i32(decx::_GPU_Vector* src, int32_t* res, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.generate_configs<_print>(src->Vec, src->Len(), S, handle);

    decx::reduce::cuda_reduce1D_u8_i32_caller_Async<uint8_t, true>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



template <bool _print>
static void decx::reduce::dev_vector_reduce_sum_fp16_fp32(decx::_GPU_Vector* src, float* res, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.generate_configs<_print>(src->Vec, src->Len(), S, handle);

    decx::reduce::cuda_reduce1D_fp16_fp32_caller_Async<de::Half, true>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Sum(de::Vector& src, double* res);


        _DECX_API_ de::DH Max(de::Vector& src, double* res);


        _DECX_API_ de::DH Sum(de::GPU_Vector& src, double* res);
    }
}


#endif