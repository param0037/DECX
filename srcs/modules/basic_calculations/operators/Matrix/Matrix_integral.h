/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_INTEGRAL_H_
#define _MATRIX_INTEGRAL_H_


#include "../../../classes/Matrix.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../scan/CUDA/scan.cuh"
#include "../../scan/CUDA/scan_caller.h"
#include "../../../../Async Engine/DecxStream/DecxStream.h"
#include "../../../../Async Engine/Async_task_threadpool/Async_Engine.h"


namespace decx
{
    namespace calc
    {
        // full scan
        template <bool _only_scan_h>
        static void cuda_matrix_integral_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode);


        static void cuda_matrix_integral_v_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode);


        template < bool _only_scan_h>
        static void cuda_matrix_integral_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode);

        // vertically scan
        static void cuda_matrix_integral_v_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode);


        template <bool _only_scan_h>
        static void cuda_matrix_integral_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode);


        static void cuda_matrix_integral_v_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode);
    }

    namespace calc
    {
        template <bool _only_scan_h>
        static void cuda_matrix_dev_integral_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode);


        static void cuda_matrix_dev_integral_v_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode);


        template <bool _only_scan_h>
        static void cuda_matrix_dev_integral_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode);


        static void cuda_matrix_dev_integral_v_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode);


        template <bool _only_scan_h>
        static void cuda_matrix_dev_integral_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode);


        static void cuda_matrix_dev_integral_v_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode);
    }
}


template <bool _only_scan_h>
static void decx::calc::cuda_matrix_integral_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<float, float>(make_uint2(src->Width(), src->Height()), S, scan_mode, true);

    decx::Ptr2D_Info<void> src_ptr2D_info = config.get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> dst_Ptr2D_info = config.get_raw_dev_ptr_dst();

    checkCudaErrors(cudaMemcpy2DAsync(src_ptr2D_info._ptr.ptr,      src_ptr2D_info._dims.x * sizeof(float),
                                      src->Mat.ptr,                 src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float), src->Height(),
                                      cudaMemcpyHostToDevice,       S->get_raw_stream_ref()));

    decx::scan::cuda_scan2D_fp32_caller_Async<_only_scan_h>(&config, S);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,                 dst->Pitch() * sizeof(float),
                                      dst_Ptr2D_info._ptr.ptr,      dst_Ptr2D_info._dims.x * sizeof(float),
                                      dst->Width() * sizeof(float), dst->Height(),
                                      cudaMemcpyDeviceToHost,       S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<float>(false);
}



template <bool _only_scan_h>
static void decx::calc::cuda_matrix_integral_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<de::Half, float>(make_uint2(src->Width(), src->Height()), S, scan_mode, true);

    decx::Ptr2D_Info<void> src_ptr2D_info = config.get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> dst_Ptr2D_info = config.get_raw_dev_ptr_dst();

    checkCudaErrors(cudaMemcpy2DAsync(src_ptr2D_info._ptr.ptr,          src_ptr2D_info._dims.x * sizeof(de::Half),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(de::Half),
                                      src->Width() * sizeof(de::Half),  src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    decx::scan::cuda_scan2D_fp16_fp32_caller_Async<_only_scan_h>(&config, S);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,                 dst->Pitch() * sizeof(float),
                                      dst_Ptr2D_info._ptr.ptr,      dst_Ptr2D_info._dims.x * sizeof(float),
                                      dst->Width() * sizeof(float), dst->Height(),
                                      cudaMemcpyDeviceToHost,       S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<de::Half>(false);
}



template <bool _only_scan_h>
static void decx::calc::cuda_matrix_integral_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<uint8_t, int>(make_uint2(src->Width(), src->Height()), S, scan_mode, true);

    decx::Ptr2D_Info<void> src_ptr2D_info = config.get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> dst_Ptr2D_info = config.get_raw_dev_ptr_dst();

    checkCudaErrors(cudaMemcpy2DAsync(src_ptr2D_info._ptr.ptr, src_ptr2D_info._dims.x * sizeof(uint8_t),
        src->Mat.ptr, src->Pitch() * sizeof(uint8_t),
        src->Width() * sizeof(uint8_t), src->Height(),
        cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::scan::cuda_scan2D_u8_i32_caller_Async<_only_scan_h>(&config, S);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,                 dst->Pitch() * sizeof(int),
                                      dst_Ptr2D_info._ptr.ptr,      dst_Ptr2D_info._dims.x * sizeof(int),
                                      dst->Width() * sizeof(int),   dst->Height(),
                                      cudaMemcpyDeviceToHost,       S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<uint8_t>(false);
}



// -------------------------------------------------- vertically scan ----------------------------------------------------------


static void decx::calc::cuda_matrix_integral_v_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<float, float>(make_uint2(src->Width(), src->Height()), S, scan_mode, false);

    decx::Ptr2D_Info<void> src_ptr2D_info = config.get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> dst_Ptr2D_info = config.get_raw_dev_ptr_dst();

    checkCudaErrors(cudaMemcpy2DAsync(src_ptr2D_info._ptr.ptr,      src_ptr2D_info._dims.x * sizeof(float),
                                      src->Mat.ptr,                 src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float), src->Height(),
                                      cudaMemcpyHostToDevice,       S->get_raw_stream_ref()));

    decx::scan::cuda_scan2D_v_fp32_caller_Async(&config, S);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,                 dst->Pitch() * sizeof(float),
                                      dst_Ptr2D_info._ptr.ptr,      dst_Ptr2D_info._dims.x * sizeof(float),
                                      dst->Width() * sizeof(float), dst->Height(),
                                      cudaMemcpyDeviceToHost,       S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<float>(false);
}



static void decx::calc::cuda_matrix_integral_v_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<de::Half, float>(make_uint2(src->Width(), src->Height()), S, scan_mode, false);

    decx::Ptr2D_Info<void> src_ptr2D_info = config.get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> dst_Ptr2D_info = config.get_raw_dev_ptr_dst();

    checkCudaErrors(cudaMemcpy2DAsync(src_ptr2D_info._ptr.ptr,          src_ptr2D_info._dims.x * sizeof(de::Half),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(de::Half),
                                      src->Width() * sizeof(de::Half),  src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    decx::scan::cuda_scan2D_v_fp16_fp32_caller_Async(&config, S);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,                 dst->Pitch() * sizeof(float),
                                      dst_Ptr2D_info._ptr.ptr,      dst_Ptr2D_info._dims.x * sizeof(float),
                                      dst->Width() * sizeof(float), dst->Height(),
                                      cudaMemcpyDeviceToHost,       S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<de::Half>(false);
}



static void decx::calc::cuda_matrix_integral_v_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<uint8_t, int>(make_uint2(src->Width(), src->Height()), S, scan_mode, false);

    decx::Ptr2D_Info<void> src_ptr2D_info = config.get_raw_dev_ptr_src();
    decx::Ptr2D_Info<void> dst_Ptr2D_info = config.get_raw_dev_ptr_dst();

    checkCudaErrors(cudaMemcpy2DAsync(src_ptr2D_info._ptr.ptr,          src_ptr2D_info._dims.x * sizeof(uint8_t),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(uint8_t),
                                      src->Width() * sizeof(uint8_t),  src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    decx::scan::cuda_scan2D_v_u8_i32_caller_Async(&config, S);

    checkCudaErrors(cudaMemcpy2DAsync(dst->Mat.ptr,                 dst->Pitch() * sizeof(int),
                                      dst_Ptr2D_info._ptr.ptr,      dst_Ptr2D_info._dims.x * sizeof(int),
                                      dst->Width() * sizeof(int), dst->Height(),
                                      cudaMemcpyDeviceToHost,       S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<uint8_t>(false);
}




template <bool _only_scan_h>
static void decx::calc::cuda_matrix_dev_integral_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<float, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, scan_mode, true);

    decx::scan::cuda_scan2D_fp32_caller_Async<_only_scan_h>(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<float>(true);
}


static void decx::calc::cuda_matrix_dev_integral_v_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<float, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, scan_mode, false);

    decx::scan::cuda_scan2D_v_fp32_caller_Async(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<de::Half>(true);
}



template <bool _only_scan_h>
static void decx::calc::cuda_matrix_dev_integral_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<de::Half, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, scan_mode, true);

    decx::scan::cuda_scan2D_fp16_fp32_caller_Async<_only_scan_h>(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<de::Half>(true);
}


static void decx::calc::cuda_matrix_dev_integral_v_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<de::Half, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, scan_mode, false);

    decx::scan::cuda_scan2D_v_fp16_fp32_caller_Async(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<de::Half>(true);
}



template <bool _only_scan_h>
static void decx::calc::cuda_matrix_dev_integral_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<uint8_t, int>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, scan_mode, true);

    decx::scan::cuda_scan2D_u8_i32_caller_Async<_only_scan_h>(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<uint8_t>(true);
}


static void decx::calc::cuda_matrix_dev_integral_v_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode)
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

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<uint8_t, int>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, scan_mode, false);

    decx::scan::cuda_scan2D_v_u8_i32_caller_Async(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    config.release_buffer<uint8_t>(true);
}




namespace decx
{
    namespace scan 
    {
        template <bool _async_call>
        void Integral2D(decx::_Matrix* src, decx::_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle, const uint32_t _stream_id = 0);


        template <bool _async_call>
        void dev_Integral2D(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle, const uint32_t _stream_id = 0);
    }
}


namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Integral(de::Matrix& src, de::Matrix& dst, const int scan2D_mode, const int scan_calc_mode);


        _DECX_API_ de::DH Integral(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int scan2D_mode, const int scan_calc_mode);


        _DECX_API_ de::DH Integral_Async(de::Matrix& src, de::Matrix& dst, const int scan2D_mode, const int scan_calc_mode, de::DecxStream& S);


        _DECX_API_ de::DH Integral_Async(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int scan2D_mode, const int scan_calc_mode, de::DecxStream& S);
    }
}


#endif