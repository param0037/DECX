/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _VECTOR_INTEGRAL_H_
#define _VECTOR_INTEGRAL_H_


#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../../Async Engine/DecxStream/DecxStream.h"
#include "../../../../Async Engine/Async_task_threadpool/Async_Engine.h"

#ifdef _DECX_CUDA_PARTS_
#include "../../scan/CUDA/scan.cuh"
#include "../../scan/CUDA/scan_caller.h"
#endif


namespace decx
{
    namespace calc 
    {
        static void cuda_vector_integral_fp32(decx::_Vector* src, decx::_Vector* dst, const int scan_mode);


        static void cuda_vector_integral_uc8(decx::_Vector* src, decx::_Vector* dst, const int scan_mode);


        static void cuda_vector_integral_fp16(decx::_Vector* src, decx::_Vector* dst, const int scan_mode);


        static void cuda_GPU_vector_integral_fp32(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, const int scan_mode);


        static void cuda_GPU_vector_integral_uc8(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, const int scan_mode);


        static void cuda_GPU_vector_integral_fp16(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, const int scan_mode);
    }
}


static void decx::calc::cuda_vector_integral_fp32(decx::_Vector* src, decx::_Vector* dst, const int scan_mode)
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

    decx::scan::cuda_scan1D_config _scan1D_config;

    _scan1D_config.generate_scan_config<4, float>(src->_length, S, scan_mode);

    checkCudaErrors(cudaMemcpyAsync(_scan1D_config.get_raw_dev_ptr_src(), src->Vec.ptr, src->_length * sizeof(float), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::scan::cuda_scan1D_fp32_caller_Async(&_scan1D_config, S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _scan1D_config.get_raw_dev_ptr_dst(), src->_length * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _scan1D_config.release_buffer<float>(false);

    S->detach();
    E->detach();
}



static void decx::calc::cuda_vector_integral_uc8(decx::_Vector* src, decx::_Vector* dst, const int scan_mode)
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

    decx::scan::cuda_scan1D_config _scan1D_config;

    _scan1D_config.generate_scan_config<8, uint8_t, int>(src->_length, S, scan_mode);

    checkCudaErrors(cudaMemcpyAsync(_scan1D_config.get_raw_dev_ptr_src(), src->Vec.ptr, src->_length * sizeof(uint8_t), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::scan::cuda_scan1D_u8_i32_caller_Async(&_scan1D_config, S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _scan1D_config.get_raw_dev_ptr_dst(), dst->_length * sizeof(int), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _scan1D_config.release_buffer<uint8_t>(false);

    S->detach();
    E->detach();
}



static void decx::calc::cuda_vector_integral_fp16(decx::_Vector* src, decx::_Vector* dst, const int scan_mode)
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

    decx::scan::cuda_scan1D_config _scan1D_config;

    _scan1D_config.generate_scan_config<8, de::Half, float>(src->_length, S, scan_mode);

    checkCudaErrors(cudaMemcpyAsync(_scan1D_config.get_raw_dev_ptr_src(), src->Vec.ptr, src->_length * sizeof(de::Half), cudaMemcpyHostToDevice, S->get_raw_stream_ref()));

    decx::scan::cuda_scan1D_fp16_caller_Async(&_scan1D_config, S);

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _scan1D_config.get_raw_dev_ptr_dst(), dst->_length * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _scan1D_config.release_buffer<de::Half>(false);

    S->detach();
    E->detach();
}



static void decx::calc::cuda_GPU_vector_integral_fp32(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, const int scan_mode)
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

    decx::scan::cuda_scan1D_config _scan1D_config;

    _scan1D_config.generate_scan_config<4, float>(src->Vec._type_cast<void>(), dst->Vec._type_cast<void>(), src->_length, S, scan_mode);

    decx::scan::cuda_scan1D_fp32_caller_Async(&_scan1D_config, S);

    E->event_record(S);
    E->synchronize();

    _scan1D_config.release_buffer<float>(true);

    S->detach();
    E->detach();
}



static void decx::calc::cuda_GPU_vector_integral_uc8(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, const int scan_mode)
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

    decx::scan::cuda_scan1D_config _scan1D_config;

    _scan1D_config.generate_scan_config<8, uint8_t, int>(src->Vec._type_cast<void>(), dst->Vec._type_cast<void>(), 
        src->_length, S, scan_mode);

    decx::scan::cuda_scan1D_u8_i32_caller_Async(&_scan1D_config, S);

    E->event_record(S);
    E->synchronize();

    _scan1D_config.release_buffer<uint8_t>(true);

    S->detach();
    E->detach();
}


static void decx::calc::cuda_GPU_vector_integral_fp16(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, const int scan_mode)
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

    decx::scan::cuda_scan1D_config _scan1D_config;

    _scan1D_config.generate_scan_config<8, de::Half, float>(src->Vec._type_cast<void>(), dst->Vec._type_cast<void>(), 
        src->_length, S, scan_mode);

    decx::scan::cuda_scan1D_fp16_caller_Async(&_scan1D_config, S);

    E->event_record(S);
    E->synchronize();

    _scan1D_config.release_buffer<de::Half>(true);

    S->detach();
    E->detach();
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Integral(de::Vector& src, de::Vector& dst, const int scan_mode);


        _DECX_API_ de::DH Integral(de::GPU_Vector& src, de::GPU_Vector& dst, const int scan_mode);


        _DECX_API_ de::DH Integral_Async(de::Vector& src, de::Vector& dst, const int scan_mode, de::DecxStream& S);


        _DECX_API_ de::DH Integral_Async(de::GPU_Vector& src, de::GPU_Vector& dst, const int scan_mode, de::DecxStream& S);
    }
}


#endif