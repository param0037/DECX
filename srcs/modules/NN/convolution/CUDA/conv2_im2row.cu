/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_MK_IM2ROW_H_
#define _CONV2_MK_IM2ROW_H_

#include "../../../classes/Tensor.h"
#include "../../../classes/TensorArray.h"
#include "fp32/conv2_mk_im2col_fp32.h"
#include "fp16/conv2_mk_im2col_fp16.h"
#include "../../../BLAS/basic_process/extension/extend_flags.h"


namespace decx
{
    namespace conv_I2R
    {
        template <bool _print>
         static void Conv2_MK_im2col_fp32(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, 
            de::GPU_Tensor& dst, const int flag, const uint2 strideXY, de::DH* handle);


         template <bool _print>
         static void Conv2_MK_im2col_fp16(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, 
            de::GPU_Tensor& dst, const int flag, const int accu_flag, const uint2 strideXY, de::DH* handle);
    }
}


template <bool _print> static void
decx::conv_I2R::Conv2_MK_im2col_fp32(de::GPU_Tensor& src, de::GPU_TensorArray& kernel, de::GPU_Tensor& dst, 
    const int flag, const uint2 strideXY, de::DH* handle)
{
    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_TensorArray* _kernel = dynamic_cast<decx::_GPU_TensorArray*>(&kernel);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);
 
    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaStreamNonBlocking);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<true>(handle);
        return;
    }

    switch (flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_NB_im2col_fp32<_print>(_src, _kernel, _dst, S, E, handle);
        }
        else {
            decx::conv_I2R::conv2_NB_im2col_fp32_stride<_print>(_src, _kernel, _dst, S, E, strideXY, handle);
        }
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_BC_im2col_fp32<_print>(_src, _kernel, _dst, S, E, handle);
        }
        else {
            decx::conv_I2R::conv2_BC_im2col_fp32_stride<_print>(_src, _kernel, _dst, S, E, strideXY, handle);
        }
        break;

    default:
        break;
    }

    S->detach();
    E->detach();
}



template <bool _print> static void 
decx::conv_I2R::Conv2_MK_im2col_fp16(de::GPU_Tensor& src, de::GPU_TensorArray& kernel,
    de::GPU_Tensor& dst, const int flag, const int accu_flag, const uint2 strideXY, de::DH* handle)
{
    decx::_GPU_Tensor* _src = dynamic_cast<decx::_GPU_Tensor*>(&src);
    decx::_GPU_TensorArray* _kernel = dynamic_cast<decx::_GPU_TensorArray*>(&kernel);
    decx::_GPU_Tensor* _dst = dynamic_cast<decx::_GPU_Tensor*>(&dst);

    decx::cuda_stream* S;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }

    decx::cuda_event* E;
    E = decx::cuda::get_cuda_event_ptr(cudaStreamNonBlocking);
    if (E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    switch (flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_NB_im2col_fp16<_print>(_src, _kernel, _dst, accu_flag, S, E, handle);
        }
        else {
            decx::conv_I2R::conv2_NB_im2col_fp16_stride<_print>(_src, _kernel, _dst, accu_flag, S, E, strideXY, handle);
        }
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_BC_im2col_fp16<_print>(_src, _kernel, _dst, accu_flag, S, E, handle);
        }
        else {
            decx::conv_I2R::conv2_BC_im2col_fp16_stride<_print>(_src, _kernel, _dst, accu_flag, S, E, strideXY, handle);
        }
        break;

    default:
        break;
    }

    S->detach();
    E->detach();
}



namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel,
            de::GPU_Tensor& dst, const de::Point2D strideXY, const int flag, const int accu_flag);
    }
}


_DECX_API_ de::DH de::cuda::Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel,
    de::GPU_Tensor& dst, const de::Point2D strideXY, const int flag, const int accu_flag)
{
    de::DH handle;
    decx::err::Success(&handle);

    if (!decx::cuda::_is_CUDA_init()) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    if (src.Type() != kernel.Type()) {
        Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
        decx::err::TypeError_NotMatch(&handle);
        return handle;
    }

    switch (src.Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv_I2R::Conv2_MK_im2col_fp16<true>(src, kernel, dst, flag, accu_flag, make_uint2(strideXY.x, strideXY.y), &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv_I2R::Conv2_MK_im2col_fp32<true>(src, kernel, dst, flag, make_uint2(strideXY.x, strideXY.y), &handle);
        break;

    default:
        break;
    }

    return handle;
}


#endif