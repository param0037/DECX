/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../large_squares/GEMM.h"
#include "VMM_callers.cuh"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"


#define _VMM_MAT_MUL_VEC_ true
#define _VMM_VEC_MUL_MAT_ false


namespace decx
{
    template <bool _is_reduce_h>
    static void _VMM_caller_fp32(decx::_Vector* vec, decx::_Matrix* mat, decx::_Vector* dst, de::DH* handle);


    template <bool _is_reduce_h>
    static void _VMM_caller_fp16(decx::_Vector* vec, decx::_Matrix* mat, decx::_Vector* dst, const uint32_t _fp16_accu, de::DH* handle);
}


template <bool _is_reduce_h>
static void decx::_VMM_caller_fp32(decx::_Vector* vec, decx::_Matrix* mat, decx::_Vector* dst, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::dot::cuda_DP2D_configs<float> _configs;
    decx::generate_VMM_config_fp32<_is_reduce_h>(&_configs, make_uint2(mat->Width(), mat->Height()), S);

    // Copy matrix data
    checkCudaErrors(cudaMemcpy2DAsync(_configs._dev_A.ptr,          _configs._dev_mat_dims.x * sizeof(float),
                                      mat->Mat.ptr,                 mat->Pitch() * sizeof(float), 
                                      mat->Width() * sizeof(float), mat->Height(), 
                                      cudaMemcpyHostToDevice,       S->get_raw_stream_ref()));
    // Copy vector data
    checkCudaErrors(cudaMemcpyAsync(_configs._dev_B.ptr, vec->Vec.ptr, vec->Len() * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const void* res_ptr = decx::_VMM_fp32_caller_async<_is_reduce_h>(&_configs, S);
    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    const uint64_t _cpy_size = (_is_reduce_h ? mat->Height() : mat->Width()) * sizeof(float);
    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, res_ptr, _cpy_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _configs.release_buffer();
}



template <bool _is_reduce_h>
static void decx::_VMM_caller_fp16(decx::_Vector* vec, decx::_Matrix* mat, decx::_Vector* dst, const uint32_t _fp16_accu, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    
    decx::dot::cuda_DP2D_configs<de::Half> _configs;
    decx::generate_VMM_config_fp16<_is_reduce_h>(&_configs, make_uint2(mat->Width(), mat->Height()), S, _fp16_accu);

    // Copy matrix data
    checkCudaErrors(cudaMemcpy2DAsync(_configs._dev_A.ptr,              _configs._dev_mat_dims.x * sizeof(de::Half),
                                      mat->Mat.ptr,                     mat->Pitch() * sizeof(de::Half),
                                      mat->Width() * sizeof(de::Half),  mat->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));
    // Copy vector data
    checkCudaErrors(cudaMemcpyAsync(_configs._dev_B.ptr, vec->Vec.ptr, vec->Len() * sizeof(de::Half), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    const void* res_ptr = decx::_VMM_fp16_caller_async<_is_reduce_h>(&_configs, S, _fp16_accu);
    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    const uint64_t _cpy_size = (_is_reduce_h ? mat->Height() : mat->Width()) * 
        (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ? sizeof(float) : sizeof(de::Half));
    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, res_ptr, _cpy_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _configs.release_buffer();
}



_DECX_API_ de::DH de::cuda::GEMM(de::Vector& A, de::Matrix& B, de::Vector& dst, const uint32_t _fp16_accu)
{
    de::DH handle;
    
    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Matrix* _B = dynamic_cast<decx::_Matrix*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::_VMM_caller_fp32<_VMM_VEC_MUL_MAT_>(_A, _B, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::_VMM_caller_fp16<_VMM_VEC_MUL_MAT_>(_A, _B, _dst, _fp16_accu, &handle);
        break;
    default:
        break;
    }
    
    return handle;
}


_DECX_API_ de::DH de::cuda::GEMM(de::Matrix& A, de::Vector& B, de::Vector& dst, const uint32_t _fp16_accu)
{
    de::DH handle;

    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    switch (_A->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::_VMM_caller_fp32<_VMM_MAT_MUL_VEC_>(_B, _A, _dst, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::_VMM_caller_fp16<_VMM_MAT_MUL_VEC_>(_B, _A, _dst, _fp16_accu, &handle);
        break;
    default:
        break;
    }

    return handle;
}