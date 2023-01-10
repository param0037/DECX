/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CONV2_MK_IM2COL_H_
#define _CONV2_MK_IM2COL_H_

#include "../../../classes/Tensor.h"
#include "../../../classes/TensorArray.h"
#include "fp32/conv2_mk_im2col_fp32.h"
#include "fp16/conv2_mk_im2col_fp16.h"

namespace decx
{
    namespace conv
    {
         static void Conv2_MK_im2col_fp32(de::Tensor& src, de::TensorArray& kernel, 
            de::Tensor& dst, const int flag, de::DH* handle);


         static void Conv2_MK_im2col_fp16(de::Tensor& src, de::TensorArray& kernel, 
            de::Tensor& dst, const int flag, const int accu_flag, de::DH* handle);
    }
}


static void decx::conv::Conv2_MK_im2col_fp32(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const int flag, de::DH* handle)
{
    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    decx::_TensorArray* _kernel = dynamic_cast<decx::_TensorArray*>(&kernel);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    decx::cuda_stream* S;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::sconv2_NB_r8_mk_im2col(_src, _kernel, _dst, S);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::sconv2_BC_r8_mk_im2col(_src, _kernel, _dst, S);
        break;

    default:
        break;
    }

    S->detach();
}



static void decx::conv::Conv2_MK_im2col_fp16(de::Tensor& src, de::TensorArray& kernel, 
    de::Tensor& dst, const int flag, const int accu_flag, de::DH* handle)
{
    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    decx::_TensorArray* _kernel = dynamic_cast<decx::_TensorArray*>(&kernel);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    decx::cuda_stream* S;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        decx::err::CUDA_Stream_access_fail(handle);
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::hconv2_NB_r8_mk_im2col(_src, _kernel, _dst, accu_flag, S);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        decx::hconv2_BC_r8_mk_im2col(_src, _kernel, _dst, accu_flag, S);
        break;

    default:
        break;
    }

    S->detach();
}



namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Conv2_multi_kernel(de::Tensor& src, de::TensorArray& kernel,
            de::Tensor& dst, const int flag, const int accu_flag);
    }
}


_DECX_API_ de::DH de::cuda::Conv2_multi_kernel(de::Tensor& src, de::TensorArray& kernel,
    de::Tensor& dst, const int flag, const int accu_flag)
{
    de::DH handle;
    decx::err::Success(&handle);

    if (!decx::cuP.is_init) {
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
        decx::conv::Conv2_MK_im2col_fp16(src, kernel, dst, flag, accu_flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Conv2_MK_im2col_fp32(src, kernel, dst, flag, &handle);
        break;

    default:
        break;
    }

    return handle;
}


#endif