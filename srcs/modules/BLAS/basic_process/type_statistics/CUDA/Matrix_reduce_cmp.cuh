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
        static void matrix_reduce2D_full_cmp_fp32(decx::_Matrix* src, float* res);

        template <bool _is_max>
        static void dev_matrix_reduce2D_full_cmp_fp32(decx::_GPU_Matrix* src, float* res);


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
    /*decx::cuda_stream* S = NULL;
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
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_dtmp1();
    
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

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_leading_ptr(), dst->Len() * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();*/

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
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S, true);
    
    //decx::Ptr2D_Info<void> _dt1 = _configs.get_dtmp1();
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
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<float> _configs;
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
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<de::Half> _configs;
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
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<de::Half> _configs;
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
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<uint8_t> _configs;
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
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<uint8_t> _configs;
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



template <bool _is_max>
static void decx::reduce::matrix_reduce2D_full_cmp_fp32(decx::_Matrix* src, float* res)
{
    /*decx::cuda_stream* S = NULL;
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

    uint32_t nominated_alloc_x = decx::utils::ceil<uint32_t>(src->Width(), 4) * 4;

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(nominated_alloc_x * src->Height(), S);

    _kp_configs.set_fill_val(((float*)src->Mat.ptr)[0]);

    checkCudaErrors(cudaMemcpy2DAsync(_kp_configs.get_dev_tmp1().ptr,   nominated_alloc_x * sizeof(float),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float),     src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    decx::reduce::reduce_cmp2D_full_fp32_Async<_is_max, false>(&_kp_configs, make_uint2(src->Width(), src->Height()), nominated_alloc_x / 4, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();*/
}




template <bool _is_max>
static void decx::reduce::dev_matrix_reduce2D_full_cmp_fp32(decx::_GPU_Matrix* src, float* res)
{
    //decx::cuda_stream* S = NULL;
    //S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    //if (S == NULL) {
    //    Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    //    return;
    //}
    //decx::cuda_event* E = NULL;
    //E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    //if (E == NULL) {
    //    Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    //    return;
    //}

    //uint32_t nominated_alloc_x = decx::utils::ceil<uint32_t>(src->Width(), 4) * 4 * 2;

    ///**
    //* For on-GPU process, the number of element that a block process of 1D is 2-times larger than
    //* that of flatten kernel (e.g. float -> 256 * 4 (1D kernel) = 2 * ((32 * 4 * 8) (flatten))
    //*/
    //decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    //_kp_configs.generate_configs(src->Mat, nominated_alloc_x * src->Height(), S);

    //float _fill_val = 0;

    //checkCudaErrors(cudaMemcpyAsync(&_fill_val, src->Mat.ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    //_kp_configs.set_fill_val(_fill_val);

    //decx::reduce::reduce_cmp2D_full_fp32_Async<_is_max, true>(&_kp_configs, make_uint2(src->Width(), src->Height()), src->Pitch() / 4, S);

    //checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    //E->event_record(S);
    //E->synchronize();

    //_kp_configs.release_buffer();
}


#endif