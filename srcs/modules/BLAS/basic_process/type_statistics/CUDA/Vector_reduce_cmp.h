/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _VECTOR_REDUCE_CMP_H_
#define _VECTOR_REDUCE_CMP_H_


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
        template <bool _print, bool _is_max>
        static void vector_reduce_max_fp32(decx::_Vector* src, float* res, de::DH* handle);


        template <bool _print, bool _is_max>
        static void vector_reduce_max_fp16(decx::_Vector* src, de::Half* res, de::DH* handle);


        template <bool _print, bool _is_max>
        static void vector_reduce_max_u8(decx::_Vector* src, uint8_t* res, de::DH* handle);
    }
}


template <bool _print, bool _is_max>
static void decx::reduce::vector_reduce_max_fp32(decx::_Vector* src, float* res, de::DH* handle)
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

    _kp_configs.set_fill_val(((float*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_dev_tmp1().ptr, src->Vec.ptr, src->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<false, _is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



template <bool _print, bool _is_max>
static void decx::reduce::vector_reduce_max_fp16(decx::_Vector* src, de::Half* res, de::DH* handle)
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

    _kp_configs.set_fill_val(((de::Half*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_dev_tmp1().ptr, src->Vec.ptr, src->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<false, _is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}




template <bool _print, bool _is_max>
static void decx::reduce::vector_reduce_max_u8(decx::_Vector* src, uint8_t* res, de::DH* handle)
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

    _kp_configs.set_fill_val(((uint8_t*)src->Vec.ptr)[0]);

    checkCudaErrors(cudaMemcpyAsync(_kp_configs.get_dev_tmp1().ptr, src->Vec.ptr, src->Len() * sizeof(uint8_t), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<false, _is_max>(&_kp_configs, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}


#endif