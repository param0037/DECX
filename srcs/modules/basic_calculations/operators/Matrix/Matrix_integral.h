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


namespace decx
{
    namespace calc
    {
        // full scan
        template <bool _print, bool _only_scan_h>
        static void cuda_matrix_integral_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print>
        static void cuda_matrix_integral_v_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print, bool _only_scan_h>
        static void cuda_matrix_integral_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle);

        // vertically scan
        template <bool _print>
        static void cuda_matrix_integral_v_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print, bool _only_scan_h>
        static void cuda_matrix_integral_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print>
        static void cuda_matrix_integral_v_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle);
    }

    namespace calc
    {
        template <bool _print, bool _only_scan_h>
        static void cuda_matrix_dev_integral_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print>
        static void cuda_matrix_dev_integral_v_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print, bool _only_scan_h>
        static void cuda_matrix_dev_integral_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print>
        static void cuda_matrix_dev_integral_v_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print, bool _only_scan_h>
        static void cuda_matrix_dev_integral_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle);


        template <bool _print>
        static void cuda_matrix_dev_integral_v_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle);
    }
}


template <bool _print, bool _only_scan_h>
static void decx::calc::cuda_matrix_integral_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }
    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, float, float>(make_uint2(src->Width(), src->Height()), S, handle, scan_mode, true);

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
}



template <bool _print, bool _only_scan_h>
static void decx::calc::cuda_matrix_integral_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, de::Half, float>(make_uint2(src->Width(), src->Height()), S, handle, scan_mode, true);

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
}



template <bool _print, bool _only_scan_h>
static void decx::calc::cuda_matrix_integral_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, uint8_t, int>(make_uint2(src->Width(), src->Height()), S, handle, scan_mode, true);

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
}



// -------------------------------------------------- vertically scan ----------------------------------------------------------


template <bool _print>
static void decx::calc::cuda_matrix_integral_v_fp32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }
    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, float, float>(make_uint2(src->Width(), src->Height()), S, handle, scan_mode, false);

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
}



template <bool _print>
static void decx::calc::cuda_matrix_integral_v_fp16(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, de::Half, float>(make_uint2(src->Width(), src->Height()), S, handle, scan_mode, false);

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
}



template <bool _print>
static void decx::calc::cuda_matrix_integral_v_uint8_i32(decx::_Matrix* src, decx::_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, uint8_t, int>(make_uint2(src->Width(), src->Height()), S, handle, scan_mode, false);

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
}




template <bool _print, bool _only_scan_h>
static void decx::calc::cuda_matrix_dev_integral_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }
    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, float, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, handle, scan_mode, true);

    decx::scan::cuda_scan2D_fp32_caller_Async<_only_scan_h>(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _print>
static void decx::calc::cuda_matrix_dev_integral_v_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle)
{
    const uint2 proc_dims = make_uint2(src->Pitch(), src->Height());
    const uint32_t pitch_v4 = src->get_layout().pitch / 4;

    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }
    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, float, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, handle, scan_mode, false);

    decx::scan::cuda_scan2D_v_fp32_caller_Async(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _print, bool _only_scan_h>
static void decx::calc::cuda_matrix_dev_integral_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, de::Half, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, handle, scan_mode, true);

    decx::scan::cuda_scan2D_fp16_fp32_caller_Async<_only_scan_h>(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _print>
static void decx::calc::cuda_matrix_dev_integral_v_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, de::Half, float>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, handle, scan_mode, false);

    decx::scan::cuda_scan2D_v_fp16_fp32_caller_Async(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _print, bool _only_scan_h>
static void decx::calc::cuda_matrix_dev_integral_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, uint8_t, int>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, handle, scan_mode, true);

    decx::scan::cuda_scan2D_u8_i32_caller_Async<_only_scan_h>(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}



template <bool _print>
static void decx::calc::cuda_matrix_dev_integral_v_uint8_i32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan_mode, de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    decx::scan::cuda_scan2D_config config;

    config.generate_scan_config<_print, uint8_t, int>(decx::Ptr2D_Info<void>(src->Mat, make_uint2(src->Pitch(), src->Height())),
        decx::Ptr2D_Info<void>(dst->Mat, make_uint2(dst->Pitch(), dst->Height())),
        make_uint2(src->Width(), src->Height()), S, handle, scan_mode, false);

    decx::scan::cuda_scan2D_v_u8_i32_caller_Async(&config, S);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
}




namespace decx
{
    namespace scan {
        void Integral2D(decx::_Matrix* src, decx::_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle);


        void dev_Integral2D(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, const int scan2D_mode, const int scan_calc_mode, de::DH* handle);
    }
}


namespace de
{
    namespace cuda {
        _DECX_API_ void Integral(de::Matrix& src, de::Matrix& dst, const int scan2D_mode, const int scan_calc_mode);


        _DECX_API_ void Integral(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int scan2D_mode, const int scan_calc_mode);
    }
}


#endif