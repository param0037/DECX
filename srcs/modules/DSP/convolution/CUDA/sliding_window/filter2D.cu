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

    decx::dsp::cuda_Filter2D_planner<float> _planner;
    _planner.plan(&src->get_layout(), &kernel->get_layout(), _extend_method, de::_FP32_, S, handle);
    Check_Runtime_Error(handle);

    const uint2 _req_dst_dims = _planner.dst_dims_req();
    dst->re_construct(src->Type(), _req_dst_dims.x, _req_dst_dims.y, S);

    _planner.run(src, kernel, dst, S, handle);
    Check_Runtime_Error(handle);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
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

    decx::dsp::cuda_Filter2D_planner<uint8_t> _planner;
    _planner.plan(&src->get_layout(), &kernel->get_layout(), _extend_method, _output_type, S, handle);
    Check_Runtime_Error(handle);

    const uint2 _req_dst_dims = _planner.dst_dims_req();
    dst->re_construct(_output_type, _req_dst_dims.x, _req_dst_dims.y, S);

    _planner.run(src, kernel, dst, S, handle);
    Check_Runtime_Error(handle);

    E->event_record(S);
    E->synchronize();

    S->detach();
    E->detach();
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

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::filter2D_u8(_src, _kernel, _dst, _extend_method, de::_UINT8_, &handle);
        break;

    default:
        break;
    }

    return handle;
}