/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

//#include "im2col_fp32.cuh"
//#include "im2col_GEMM_fp32.cuh"
//#include "../../../../classes/GPU_Tensor.h"
//#include "../../../../classes/GPU_TensorArray.h"
//#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"

#include "cuda_conv2D_fp32_im2col_planner.cuh"


namespace de
{
    namespace nn {
        namespace cuda
        {
            _DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst,
                const de::Point2D strides = { 1, 1 }, const const de::extend_label extend = de::extend_label::_EXTEND_NONE_);
        }
    }
}


namespace decx
{
    namespace nn {
        static void conv2D_im2col_fp32_caller(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst,
            const uint2 strides, const de::extend_label extend, de::DH* handle);
    }
}


static void _CRSR_
decx::nn::conv2D_im2col_fp32_caller(decx::_GPU_Tensor* src, 
                                    decx::_GPU_TensorArray* kernel, 
                                    decx::_GPU_Tensor* dst,
                                    const uint2 strides, 
                                    const de::extend_label extend,
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

    decx::nn::cuda_conv2D_fp32_im2col_planner planner;
    planner.plan(&src->get_layout(), kernel, extend, make_uint2(strides.x, strides.y), S, handle);
    Check_Runtime_Error(handle);

    const uint3 dst_dims = planner.dst_dims_query();
    dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, dst_dims.y, dst_dims.z, dst_dims.x);
    
    planner.run(src, kernel, dst, S, handle);

    E->event_record(S);
    E->synchronize();



    E->detach();
    S->detach();
}


_DECX_API_ de::DH de::nn::cuda::Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst,
    const de::Point2D strides, const de::extend_label extend)
{
    de::DH handle;

    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_TensorArray* _kernel = dynamic_cast<decx::_GPU_TensorArray*>(&kernel);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);
    
    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::nn::conv2D_im2col_fp32_caller(_src, _kernel, _dst, make_uint2(strides.x, strides.y), extend, &handle);
        break;

    default:
        break;
    }
    
    return handle;
}