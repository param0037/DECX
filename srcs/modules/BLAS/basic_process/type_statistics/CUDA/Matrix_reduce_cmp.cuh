/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_REDUCE_CMP_CUH_
#define _MATRIX_REDUCE_CMP_CUH_


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
        template <bool _is_max>
        static void matrix_reduce2D_full_cmp_fp32(decx::_Matrix* src, de::DecxNumber* res);
        template <bool _is_max>
        static void dev_matrix_reduce2D_full_cmp_fp32(decx::_GPU_Matrix* src, de::DecxNumber* res);

        template <bool _is_max>
        static void matrix_reduce2D_full_cmp_int32(decx::_Matrix* src, de::DecxNumber* res);
        template <bool _is_max>
        static void dev_matrix_reduce2D_full_cmp_int32(decx::_GPU_Matrix* src, de::DecxNumber* res);

        template <bool _is_max>
        static void matrix_reduce2D_full_cmp_fp16(decx::_Matrix* src, de::DecxNumber* res);
        template <bool _is_max>
        static void dev_matrix_reduce2D_full_cmp_fp16(decx::_GPU_Matrix* src, de::DecxNumber* res);

        template <bool _is_max>
        static void matrix_reduce2D_full_cmp_u8(decx::_Matrix* src, de::DecxNumber* res);
        template <bool _is_max>
        static void dev_matrix_reduce2D_full_cmp_u8(decx::_GPU_Matrix* src, de::DecxNumber* res);

        template <bool _is_max>
        static void matrix_reduce2D_full_cmp_fp64(decx::_Matrix* src, de::DecxNumber* res);
        template <bool _is_max>
        static void dev_matrix_reduce2D_full_cmp_fp64(decx::_GPU_Matrix* src, de::DecxNumber* res);

        // 1-way
        template <bool _is_max, bool _is_reduce_h>
        static void matrix_reduce2D_1way_cmp_fp32(decx::_Matrix* src, decx::_Vector* dst);
        template <bool _is_max, bool _is_reduce_h>
        static void dev_matrix_reduce2D_1way_cmp_fp32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst);

        template <bool _is_max, bool _is_reduce_h>
        static void matrix_reduce2D_1way_cmp_fp16(decx::_Matrix* src, decx::_Vector* dst);
        template <bool _is_max, bool _is_reduce_h>
        static void dev_matrix_reduce2D_1way_cmp_fp16(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst);

        template <bool _is_max, bool _is_reduce_h>
        static void matrix_reduce2D_1way_cmp_u8(decx::_Matrix* src, decx::_Vector* dst);
        template <bool _is_max, bool _is_reduce_h>
        static void dev_matrix_reduce2D_1way_cmp_u8(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst);
    }
}




template <bool _is_max, bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_cmp_fp32(decx::_Matrix* src, decx::_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<float> _configs;
    _configs.set_cmp_or_not(true);
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S, true);
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_src();
    
    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                    _dt1._dims.x * sizeof(float),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float),     src->Height(),               
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_cmp2D_h_fp32_Async<_is_max>(&_configs, S);
    }
    else {
        decx::reduce::reduce_cmp2D_v_fp32_Async<_is_max>(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_dst(), dst->Len() * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}




template <bool _is_max, bool _is_reduce_h>
static void decx::reduce::dev_matrix_reduce2D_1way_cmp_fp32(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<float> _configs;
    _configs.set_cmp_or_not(true);
    _configs.generate_configs<_is_reduce_h>(src->Mat, dst->Vec.ptr, src->Pitch(), make_uint2(src->Width(), src->Height()), S, true);
    
    if (_is_reduce_h) {
        decx::reduce::reduce_cmp2D_h_fp32_Async<_is_max>(&_configs, S);
    }
    else {
        decx::reduce::reduce_cmp2D_v_fp32_Async<_is_max>(&_configs, S);
    }

    E->event_record(S);
    E->synchronize();
}




template <bool _is_max, bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_cmp_fp16(decx::_Matrix* src, decx::_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<de::Half> _configs;
    _configs.set_cmp_or_not(true);
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S, true);
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_src();
    
    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                    _dt1._dims.x * sizeof(de::Half),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(de::Half),
                                      src->Width() * sizeof(de::Half),  src->Height(),               
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_cmp2D_h_fp16_Async<_is_max>(&_configs, S);
    }
    else {
        decx::reduce::reduce_cmp2D_v_fp16_Async<_is_max>(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_dst(), dst->Len() * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}




template <bool _is_max, bool _is_reduce_h>
static void decx::reduce::dev_matrix_reduce2D_1way_cmp_fp16(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<de::Half> _configs;
    _configs.set_cmp_or_not(true);
    _configs.generate_configs<_is_reduce_h>(src->Mat, dst->Vec.ptr, src->Pitch(), make_uint2(src->Width(), src->Height()), S, true);
    
    if (_is_reduce_h) {
        decx::reduce::reduce_cmp2D_h_fp16_Async<_is_max>(&_configs, S);
    }
    else {
        decx::reduce::reduce_cmp2D_v_fp16_Async<_is_max>(&_configs, S);
    }

    E->event_record(S);
    E->synchronize();
}



template <bool _is_max, bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_cmp_u8(decx::_Matrix* src, decx::_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<uint8_t> _configs;
    _configs.set_cmp_or_not(true);
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S, true);
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_src();
    
    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                    _dt1._dims.x * sizeof(uint8_t),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(uint8_t),
                                      src->Width() * sizeof(uint8_t),   src->Height(),               
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_cmp2D_h_u8_Async<_is_max>(&_configs, S);
    }
    else {
        decx::reduce::reduce_cmp2D_v_u8_Async<_is_max>(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_dst(), dst->Len() * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}




template <bool _is_max, bool _is_reduce_h>
static void decx::reduce::dev_matrix_reduce2D_1way_cmp_u8(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<uint8_t> _configs;
    _configs.set_cmp_or_not(true);
    _configs.generate_configs<_is_reduce_h>(src->Mat, dst->Vec.ptr, src->Pitch(), make_uint2(src->Width(), src->Height()), S, true);
    
    if (_is_reduce_h) {
        decx::reduce::reduce_cmp2D_h_u8_Async<_is_max>(&_configs, S);
    }
    else {
        decx::reduce::reduce_cmp2D_v_u8_Async<_is_max>(&_configs, S);
    }

    E->event_record(S);
    E->synchronize();
}


// ------------------------------------------------ full ------------------------------------------------------

template <bool _is_max>
static void decx::reduce::matrix_reduce2D_full_cmp_fp32(decx::_Matrix* src, de::DecxNumber* res)
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
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<float, float>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);

    // set the filling value
    _kp_configs.set_fill_val(((float*)src->Mat.ptr)[0]);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_fp32_Async<_is_max>(&_kp_configs, _d_src.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::matrix_reduce2D_full_cmp_int32(decx::_Matrix* src, de::DecxNumber* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), _CU_REDUCE1D_MEM_ALIGN_4B_) * _CU_REDUCE1D_MEM_ALIGN_4B_, 
                                        src->Height());
    if (decx::alloc::_device_malloc(&_d_src, alloc_dims.x * alloc_dims.y * sizeof(int32_t), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    // Transfer data from host to deivce
    checkCudaErrors(cudaMemcpy2DAsync(_d_src.ptr,                       alloc_dims.x * sizeof(int32_t),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(int32_t),
                                      src->Width() * sizeof(int32_t),   src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<int32_t> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<int32_t, int32_t>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);

    // set the filling value
    _kp_configs.set_fill_val(((int32_t*)src->Mat.ptr)[0]);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_int32_Async<_is_max>(&_kp_configs, _d_src.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_INT32_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}





template <bool _is_max>
static void decx::reduce::matrix_reduce2D_full_cmp_fp16(decx::_Matrix* src, de::DecxNumber* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), _CU_REDUCE1D_MEM_ALIGN_2B_) * _CU_REDUCE1D_MEM_ALIGN_2B_, 
                                        src->Height());
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

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<de::Half, de::Half>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);

    // set the filling value
    _kp_configs.set_fill_val(((de::Half*)src->Mat.ptr)[0]);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_fp16_Async<_is_max>(&_kp_configs, _d_src.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::matrix_reduce2D_full_cmp_u8(decx::_Matrix* src, de::DecxNumber* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), _CU_REDUCE1D_MEM_ALIGN_1B_) * _CU_REDUCE1D_MEM_ALIGN_1B_, 
                                        src->Height());
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

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<uint8_t, uint8_t>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);

    // set the filling value
    _kp_configs.set_fill_val(((uint8_t*)src->Mat.ptr)[0]);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_u8_Async<_is_max>(&_kp_configs, _d_src.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_UINT8_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::matrix_reduce2D_full_cmp_fp64(decx::_Matrix* src, de::DecxNumber* res)
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

    decx::PtrInfo<void> _d_src;
    const uint2 alloc_dims = make_uint2(decx::utils::ceil<uint32_t>(src->Width(), 4) * 4, src->Height());
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
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<double, double>(&_kp_configs, alloc_dims.x, proc_dims_v1, S);

    // set the filling value
    _kp_configs.set_fill_val(((double*)src->Mat.ptr)[0]);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_fp64_Async<_is_max>(&_kp_configs, _d_src.ptr, proc_dims_v1, /*alloc_dims.x,*/ S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP64_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}



// ----------------------------------------------- dev_ ---------------------------------------------

template <bool _is_max>
static void decx::reduce::dev_matrix_reduce2D_full_cmp_fp32(decx::_GPU_Matrix* src, de::DecxNumber* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<float, float>(&_kp_configs, src->Pitch(), proc_dims_v1, S);

    float _fill_val = 0;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Mat.ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    _kp_configs.set_fill_val(_fill_val);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_fp32_Async<_is_max>(&_kp_configs, src->Mat.ptr, proc_dims_v1, /*src->Pitch(),*/ S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::dev_matrix_reduce2D_full_cmp_int32(decx::_GPU_Matrix* src, de::DecxNumber* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<int32_t> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<int32_t, int32_t>(&_kp_configs, src->Pitch(), proc_dims_v1, S);

    int32_t _fill_val = 0;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Mat.ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    _kp_configs.set_fill_val(_fill_val);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_int32_Async<_is_max>(&_kp_configs, src->Mat.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_INT32_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}





template <bool _is_max>
static void decx::reduce::dev_matrix_reduce2D_full_cmp_fp16(decx::_GPU_Matrix* src, de::DecxNumber* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<de::Half> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<de::Half, de::Half>(&_kp_configs, src->Pitch(), proc_dims_v1, S);

    de::Half _fill_val;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Mat.ptr, 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    _kp_configs.set_fill_val(_fill_val);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_fp16_Async<_is_max>(&_kp_configs, src->Mat.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(de::Half), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




template <bool _is_max>
static void decx::reduce::dev_matrix_reduce2D_full_cmp_u8(decx::_GPU_Matrix* src, de::DecxNumber* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<uint8_t> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<uint8_t, uint8_t>(&_kp_configs, src->Pitch(), proc_dims_v1, S);

    uint8_t _fill_val;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Mat.ptr, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    _kp_configs.set_fill_val(_fill_val);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_u8_Async<_is_max>(&_kp_configs, src->Mat.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_UINT8_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}





template <bool _is_max>
static void decx::reduce::dev_matrix_reduce2D_full_cmp_fp64(decx::_GPU_Matrix* src, de::DecxNumber* res)
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

    const uint2 proc_dims_v1 = make_uint2(src->Width(), src->Height());

    decx::reduce::cuda_reduce1D_configs<double> _kp_configs;
    _kp_configs.set_cmp_or_not(true);

    // Generate the configs for postprocessing of 1D reduction
    // Obtain whether only one flatten kernel is OK
    const bool _more_than_flatten = decx::reduce::reduce2D_flatten_postproc_configs_gen<double, double>(&_kp_configs, src->Pitch(), proc_dims_v1, S);

    double _fill_val;
    checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Mat.ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    _kp_configs.set_fill_val(_fill_val);

    // Call the kernels
    // Obtain the pointer where the final value is stored
    const void* _dst_ptr = decx::reduce::reduce_cmp2D_full_fp64_Async<_is_max>(&_kp_configs, src->Mat.ptr, proc_dims_v1, S, _more_than_flatten);

    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), _dst_ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP64_);

    E->event_record(S);
    E->synchronize();

    // Release the buffers so that the configs can be safely destructed
    _kp_configs.release_buffer();
}




#endif