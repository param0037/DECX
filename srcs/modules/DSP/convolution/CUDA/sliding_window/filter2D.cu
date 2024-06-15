/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "filter2D.cuh"
#include "cuda_Filter2D_planner.cuh"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"


namespace decx
{
    namespace dsp {
        static void filter2D_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
            const de::extend_label _extend_method, de::DH* handle);


        static void filter2D_fp64(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
            const de::extend_label _extend_method, de::DH* handle);


        static void filter2D_u8(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
            const de::extend_label _extend_method, const de::_DATA_TYPES_FLAGS_ _output_type, de::DH* handle);
    }
}


static void decx::dsp::filter2D_fp32(decx::_GPU_Matrix* src, 
                          decx::_GPU_Matrix* kernel, 
                          decx::_GPU_Matrix* dst,
                          const de::extend_label _extend_method, 
                          de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (!decx::dsp::cuda_Filter2D_planner<float>::validate_kerW(kernel->Width())) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ConvBadKernel,
            CU_FILTER2D_KERNEL_OVERRANGED);
        return;
    }

    if (decx::dsp::_cuda_filter2D_fp32._res_ptr == NULL) {
        decx::dsp::_cuda_filter2D_fp32.RegisterResource(new decx::dsp::cuda_Filter2D_planner<float>, 5,
            &decx::dsp::cuda_Filter2D_planner<float>::release);
    }

    decx::dsp::_cuda_filter2D_fp32.lock();

    decx::dsp::cuda_Filter2D_planner<float>* _planner =
        decx::dsp::_cuda_filter2D_fp32.get_resource_raw_ptr<decx::dsp::cuda_Filter2D_planner<float>>();

    if (_planner->changed(&src->get_layout(), &kernel->get_layout(),
        _extend_method, de::_FP32_)) {
        _planner->plan(&src->get_layout(), &kernel->get_layout(), _extend_method, de::_FP32_, S, handle);
        Check_Runtime_Error(handle);
    }

    const uint2 _req_dst_dims = _planner->dst_dims_req();
    dst->re_construct(src->Type(), _req_dst_dims.x, _req_dst_dims.y, S);

    _planner->run(src, kernel, dst, S, handle);
    Check_Runtime_Error(handle);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::dsp::_cuda_filter2D_fp32.unlock();
}



static void decx::dsp::filter2D_fp64(decx::_GPU_Matrix* src, 
                          decx::_GPU_Matrix* kernel, 
                          decx::_GPU_Matrix* dst,
                          const de::extend_label _extend_method, 
                          de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (!decx::dsp::cuda_Filter2D_planner<double>::validate_kerW(kernel->Width())) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ConvBadKernel,
            CU_FILTER2D_KERNEL_OVERRANGED);
        return;
    }

    if (decx::dsp::_cuda_filter2D_fp64._res_ptr == NULL) {
        decx::dsp::_cuda_filter2D_fp64.RegisterResource(new decx::dsp::cuda_Filter2D_planner<double>, 5,
            &decx::dsp::cuda_Filter2D_planner<double>::release);
    }

    decx::dsp::_cuda_filter2D_fp64.lock();

    decx::dsp::cuda_Filter2D_planner<double>* _planner =
        decx::dsp::_cuda_filter2D_fp64.get_resource_raw_ptr<decx::dsp::cuda_Filter2D_planner<double>>();

    if (_planner->changed(&src->get_layout(), &kernel->get_layout(),
        _extend_method, de::_FP64_)) {
        _planner->plan(&src->get_layout(), &kernel->get_layout(), _extend_method, de::_FP64_, S, handle);
        Check_Runtime_Error(handle);
    }

    const uint2 _req_dst_dims = _planner->dst_dims_req();
    dst->re_construct(src->Type(), _req_dst_dims.x, _req_dst_dims.y, S);

    _planner->run(src, kernel, dst, S, handle);
    Check_Runtime_Error(handle);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::dsp::_cuda_filter2D_fp64.unlock();
}


static void decx::dsp::filter2D_u8(decx::_GPU_Matrix* src, 
                        decx::_GPU_Matrix* kernel, 
                        decx::_GPU_Matrix* dst,
                        const de::extend_label _extend_method, 
                        const de::_DATA_TYPES_FLAGS_ _output_type, 
                        de::DH* handle)
{
    decx::cuda_stream* S = NULL;
    decx::cuda_event* E = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM,
            CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    if (E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT,
            CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    if (!decx::dsp::cuda_Filter2D_planner<uint8_t>::validate_kerW(kernel->Width())) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ConvBadKernel,
            CU_FILTER2D_KERNEL_OVERRANGED);
        return;
    }

    if (decx::dsp::_cuda_filter2D_u8._res_ptr == NULL) {
        decx::dsp::_cuda_filter2D_u8.RegisterResource(new decx::dsp::cuda_Filter2D_planner<uint8_t>, 5,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::release);
    }

    decx::dsp::_cuda_filter2D_u8.lock();

    decx::dsp::cuda_Filter2D_planner<uint8_t>* _planner =
        decx::dsp::_cuda_filter2D_u8.get_resource_raw_ptr<decx::dsp::cuda_Filter2D_planner<uint8_t>>();

    if (_planner->changed(&src->get_layout(), &kernel->get_layout(),
        _extend_method, _output_type)) {
        _planner->plan(&src->get_layout(), &kernel->get_layout(), _extend_method, _output_type, S, handle);
        Check_Runtime_Error(handle);
    }

    const uint2 _req_dst_dims = _planner->dst_dims_req();
    dst->re_construct(src->Type(), _req_dst_dims.x, _req_dst_dims.y, S);

    _planner->run(src, kernel, dst, S, handle);
    Check_Runtime_Error(handle);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();

    decx::dsp::_cuda_filter2D_u8.unlock();
}


_DECX_API_ de::DH de::dsp::cuda::Filter2D(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
    const de::extend_label _extend_method)
{
    de::DH handle;

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _kernel = dynamic_cast<decx::_GPU_Matrix*>(&kernel);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::filter2D_fp32(_src, _kernel, _dst, _extend_method, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::dsp::filter2D_fp64(_src, _kernel, _dst, _extend_method, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::filter2D_u8(_src, _kernel, _dst, _extend_method, de::_UINT8_, &handle);
        break;

    default:
        break;
    }

    return handle;
}
