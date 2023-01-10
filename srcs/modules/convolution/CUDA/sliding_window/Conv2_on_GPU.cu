/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "../../../classes/GPU_Matrix.h"
#include "fp32/dev_conv2_border_const_fp32.h"
#include "fp32/dev_conv2_border_ignored_fp32.h"
#include "fp16/dev_conv2_border_const_fp16.h"
#include "fp16/dev_conv2_border_ignored_fp16.h"


namespace decx
{
    namespace cuda
    {
        void dev_Conv2_on_GPU_fp32(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst, const uint flag, de::DH* handle);


        void dev_Conv2_on_GPU_fp16(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
            const uint conv_flag, const int accu_flag, de::DH* handle);
    }
}



void decx::cuda::dev_Conv2_on_GPU_fp32(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst, const uint flag, de::DH* handle)
{
    decx::_GPU_Matrix& _src = dynamic_cast<decx::_GPU_Matrix&>(src);
    decx::_GPU_Matrix& _kernel = dynamic_cast<decx::_GPU_Matrix&>(kernel);
    decx::_GPU_Matrix& _dst = dynamic_cast<decx::_GPU_Matrix&>(dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_sConv2_border_ignore(_src, _kernel, _dst, handle);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_sConv2_border_zero(_src, _kernel, _dst, handle);
        break;
    default:
        break;
    }
}



void decx::cuda::dev_Conv2_on_GPU_fp16(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
    const uint conv_flag, const int accu_flag, de::DH* handle)
{
    decx::_GPU_Matrix& _src = dynamic_cast<decx::_GPU_Matrix&>(src);
    decx::_GPU_Matrix& _kernel = dynamic_cast<decx::_GPU_Matrix&>(kernel);
    decx::_GPU_Matrix& _dst = dynamic_cast<decx::_GPU_Matrix&>(dst);

    switch (conv_flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::dev_hConv2_border_ignore(_src, _kernel, _dst, handle, accu_flag);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::dev_hConv2_border_zero(_src, _kernel, _dst, handle, accu_flag);
        break;
    default:
        break;
    }
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Conv2(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst, const uint conv_flag, const int accu_flag);
    }
}



_DECX_API_
de::DH de::cuda::Conv2(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
    const uint conv_flag, const int accu_flag)
{
    de::DH handle;
    decx::err::Success(&handle);

    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CUDA_NOT_INIT);
        decx::err::CUDA_Not_init(&handle);
        return handle;
    }

    switch (src.Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cuda::dev_Conv2_on_GPU_fp32(src, kernel, dst, conv_flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::dev_Conv2_on_GPU_fp16(src, kernel, dst, conv_flag, accu_flag, &handle);
        break;
    default:
        break;
    }
    return handle;
}