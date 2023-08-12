/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_REDUCE_SUM_CUH_
#define _MATRIX_REDUCE_SUM_CUH_


#include "../Matrix_reduce.h"
#include "../../float_half_convert.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_Vector.h"


namespace decx
{
    namespace reduce
    {
        static void matrix_reduce2D_full_sum_fp32(decx::_Matrix* src, float* res);
        static void dev_matrix_reduce2D_full_sum_fp32(decx::_GPU_Matrix* src, float* res);

        static void matrix_reduce2D_full_sum_fp16(decx::_Matrix* src, float* res);
        static void dev_matrix_reduce2D_full_sum_fp16(decx::_GPU_Matrix* src, float* res);

        static void matrix_reduce2D_full_sum_u8_i32(decx::_Matrix* src, int32_t* res);
        static void dev_matrix_reduce2D_full_sum_u8_i32(decx::_GPU_Matrix* src, int32_t* res);

        static void matrix_reduce2D_full_sum_fp64(decx::_Matrix* src, double* res);
        static void dev_matrix_reduce2D_full_sum_fp64(decx::_GPU_Matrix* src, double* res);

        static void matrix_reduce2D_full_sum_i32(decx::_Matrix* src, int32_t* res);
        static void dev_matrix_reduce2D_full_sum_i32(decx::_GPU_Matrix* src, int32_t* res);


        template <bool _is_reduce_h>
        static void matrix_reduce2D_1way_sum_fp32(decx::_Matrix* src, decx::_Vector* dst);
        template <bool _is_reduce_h>
        static void dev_matrix_reduce2D_1way_sum_fp32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst);

        template <bool _is_reduce_h>
        static void matrix_reduce2D_1way_sum_fp16_fp32(decx::_Matrix* src, decx::_Vector* dst);
        template <bool _is_reduce_h>
        static void dev_matrix_reduce2D_1way_sum_fp16_fp32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst);

        template <bool _is_reduce_h>
        static void matrix_reduce2D_1way_sum_u8_i32(decx::_Matrix* src, decx::_Vector* dst);
        template <bool _is_reduce_h>
        static void dev_matrix_reduce2D_1way_sum_u8_i32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst);
    }
}


template <bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_sum_fp32(decx::_Matrix* src, decx::_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<float> _configs;
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S);
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_src();
    
    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                    _dt1._dims.x * sizeof(float),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float),     src->Height(),               
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_fp32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_fp32_Async(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_dst(), dst->Len() * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



template <bool _is_reduce_h>
static void decx::reduce::dev_matrix_reduce2D_1way_sum_fp32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }

    decx::reduce::cuda_reduce2D_1way_configs<float> _configs;
    _configs.generate_configs<_is_reduce_h>(src->Mat, dst->Vec.ptr, src->Pitch(), make_uint2(src->Width(), src->Height()), S);

    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_fp32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_fp32_Async(&_configs, S);
    }

    E->event_record(S);
    E->synchronize();
}


template <bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_sum_fp16_fp32(decx::_Matrix* src, decx::_Vector* dst)
{
    using namespace std;
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<de::Half> _configs;
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S);
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_src();
    
    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                    _dt1._dims.x * sizeof(de::Half),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(de::Half),
                                      src->Width() * sizeof(de::Half),  src->Height(),               
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_fp16_fp32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_fp16_fp32_Async(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_dst(), dst->Len() * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



template <bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_sum_u8_i32(decx::_Matrix* src, decx::_Vector* dst)
{
    using namespace std;
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }

    decx::reduce::cuda_reduce2D_1way_configs<uint8_t> _configs;
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S);

    decx::Ptr2D_Info<void> _dt1 = _configs.get_src();

    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                        _dt1._dims.x * sizeof(uint8_t),
                                      src->Mat.ptr,                         src->Pitch() * sizeof(uint8_t),
                                      src->Width() * sizeof(uint8_t),       src->Height(),
                                      cudaMemcpyHostToDevice,               S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_u8_i32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_u8_i32_Async(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_dst(), dst->Len() * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



template <bool _is_reduce_h>
static void decx::reduce::dev_matrix_reduce2D_1way_sum_u8_i32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }

    decx::reduce::cuda_reduce2D_1way_configs<uint8_t> _configs;
    _configs.generate_configs<_is_reduce_h>(src->Mat, dst->Vec.ptr, src->Pitch(), make_uint2(src->Width(), src->Height()), S);

    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_u8_i32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_u8_i32_Async(&_configs, S);
    }

    E->event_record(S);
    E->synchronize();
}


template <bool _is_reduce_h>
static void decx::reduce::dev_matrix_reduce2D_1way_sum_fp16_fp32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst)
{
    using namespace std;
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<de::Half> _configs;
    _configs.generate_configs<_is_reduce_h>(src->Mat, dst->Vec.ptr, src->Pitch(), make_uint2(src->Width(), src->Height()), S);
    
    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_fp16_fp32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_fp16_fp32_Async(&_configs, S);
    }

    E->event_record(S);
    E->synchronize();
}


// ------------------------------------------- full ----------------------------------------

static void decx::reduce::matrix_reduce2D_full_sum_fp32(decx::_Matrix* src, float* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), 4) * 4, src->Height());
    if (decx::alloc::_device_malloc(&_d_src, alloc_dims.x * alloc_dims.y * sizeof(float), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Transfer data from host to deivce
    checkCudaErrors(cudaMemcpy2DAsync(_d_src.ptr,                   alloc_dims.x * sizeof(float),
                                      src->Mat.ptr,                 src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float), src->Height(),
                                      cudaMemcpyHostToDevice,       S->get_raw_stream_ref()));

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<float>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_fp32_Async(&_kp_configs, _d_src.ptr, proc_dims_v1, alloc_dims.x, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}



static void decx::reduce::matrix_reduce2D_full_sum_fp16(decx::_Matrix* src, float* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), 8) * 8, src->Height());
    if (decx::alloc::_device_malloc(&_d_src, alloc_dims.x * alloc_dims.y * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Transfer data from host to deivce
    checkCudaErrors(cudaMemcpy2DAsync(_d_src.ptr,                       alloc_dims.x * sizeof(de::Half),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(de::Half),
                                      src->Width() * sizeof(de::Half),  src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<de::Half, float>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_fp16_fp32_Async(&_kp_configs, _d_src.ptr, proc_dims_v1, alloc_dims.x, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




static void decx::reduce::matrix_reduce2D_full_sum_u8_i32(decx::_Matrix* src, int32_t* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), 16) * 16, src->Height());
    if (decx::alloc::_device_malloc(&_d_src, alloc_dims.x * alloc_dims.y * sizeof(uint8_t), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Transfer data from host to deivce
    checkCudaErrors(cudaMemcpy2DAsync(_d_src.ptr,                       alloc_dims.x * sizeof(uint8_t),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(uint8_t),
                                      src->Width() * sizeof(uint8_t),   src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<int32_t> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<uint8_t, int32_t>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_u8_i32_Async(&_kp_configs, _d_src.ptr, proc_dims_v1, alloc_dims.x, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




static void decx::reduce::dev_matrix_reduce2D_full_sum_fp32(decx::_GPU_Matrix* src, float* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<float, float>(&_kp_configs, src->Pitch(), proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_fp32_Async(&_kp_configs, src->Mat.ptr, proc_dims_v1, src->Pitch(), S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




static void decx::reduce::matrix_reduce2D_full_sum_fp64(decx::_Matrix* src, double* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), 2) * 2, src->Height());
    if (decx::alloc::_device_malloc(&_d_src, alloc_dims.x * alloc_dims.y * sizeof(double), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Transfer data from host to deivce
    checkCudaErrors(cudaMemcpy2DAsync(_d_src.ptr,                       alloc_dims.x * sizeof(double),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(double),
                                      src->Width() * sizeof(double),    src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<double> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<double, double>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_fp64_Async(&_kp_configs, _d_src.ptr, proc_dims_v1, alloc_dims.x, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}



static void decx::reduce::dev_matrix_reduce2D_full_sum_fp16(decx::_GPU_Matrix* src, float* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<de::Half, float>(&_kp_configs, src->Pitch(), proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_fp16_fp32_Async(&_kp_configs, src->Mat.ptr, proc_dims_v1, src->Pitch(), S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




static void decx::reduce::matrix_reduce2D_full_sum_i32(decx::_Matrix* src, int32_t* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), 4) * 4, src->Height());
    if (decx::alloc::_device_malloc(&_d_src, alloc_dims.x * alloc_dims.y * sizeof(int32_t), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Transfer data from host to deivce
    checkCudaErrors(cudaMemcpy2DAsync(_d_src.ptr,                           alloc_dims.x * sizeof(int32_t),
                                      src->Mat.ptr,                         src->Pitch() * sizeof(int32_t),
                                      src->Width() * sizeof(int32_t),       src->Height(),
                                      cudaMemcpyHostToDevice,               S->get_raw_stream_ref()));

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<int32_t> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<int32_t, int32_t>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_i32_Async(&_kp_configs, _d_src.ptr, proc_dims_v1, alloc_dims.x, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}



static void decx::reduce::dev_matrix_reduce2D_full_sum_u8_i32(decx::_GPU_Matrix* src, int32_t* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<int32_t> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<uint8_t, int32_t>(&_kp_configs, src->Pitch(), proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_u8_i32_Async(&_kp_configs, src->Mat.ptr, proc_dims_v1, src->Pitch(), S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




static void decx::reduce::dev_matrix_reduce2D_full_sum_fp64(decx::_GPU_Matrix* src, double* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());
    
    decx::reduce::cuda_reduce1D_configs<double> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<double, double>(&_kp_configs, src->Pitch(), proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_fp64_Async(&_kp_configs, src->Mat.ptr, proc_dims_v1, src->Pitch(), S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




static void decx::reduce::dev_matrix_reduce2D_full_sum_i32(decx::_GPU_Matrix* src, int32_t* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<int32_t> _kp_configs;
    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<int32_t, int32_t>(&_kp_configs, src->Pitch(), proc_dims_v1, S);
    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_sum2D_full_i32_Async(&_kp_configs, src->Mat.ptr, proc_dims_v1, src->Pitch(), S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res, _dst_ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}


#endif