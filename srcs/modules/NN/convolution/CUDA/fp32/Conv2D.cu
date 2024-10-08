/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


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

    if (decx::nn::_conv2_fp32_planner == NULL) {
        decx::nn::_conv2_fp32_planner = new decx::nn::cuda_conv2D_fp32_im2col_planner;
    }
    if (decx::nn::_conv2_fp32_planner->changed(&src->get_layout(), kernel, extend, make_uint2(strides.x, strides.y))) {
        decx::nn::_conv2_fp32_planner->plan(&src->get_layout(), kernel, extend, make_uint2(strides.x, strides.y), S, handle);
        Check_Runtime_Error(handle);
    }

    const uint3 dst_dims = decx::nn::_conv2_fp32_planner->dst_dims_query();
    dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, dst_dims.y, dst_dims.z, dst_dims.x, S);

    decx::nn::_conv2_fp32_planner->update_dst_layout(&dst->get_layout());

    decx::nn::_conv2_fp32_planner->run(src, kernel, dst, S, handle);

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
        //printf("yes\n");
        decx::nn::conv2D_im2col_fp32_caller(_src, _kernel, _dst, make_uint2(strides.x, strides.y), extend, &handle);
        break;

    default:
        break;
    }
    
    return handle;
}
