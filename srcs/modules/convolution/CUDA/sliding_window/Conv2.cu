/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "../../../classes/MatrixArray.h"
#include "../../../classes/GPU_Matrix.h"
#include "fp32/conv2_border_ignored_fp32.h"
#include "fp16/conv2_border_ignore_fp16.h"
#include "fp32/conv2_border_const_fp32.h"
#include "fp16/conv2_border_const_fp16.h"



namespace decx
{
    namespace cuda
    {
        void Conv2_fp32(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint flag, de::DH* handle);
        

        void Conv2_fp16(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint conv_flag, const int accu_flag, de::DH* handle);
    }
}




void decx::cuda::Conv2_fp32(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint flag, 
    de::DH* handle)
{
    decx::_Matrix& _src = dynamic_cast<decx::_Matrix&>(src);
    decx::_Matrix& _kernel = dynamic_cast<decx::_Matrix&>(kernel);
    decx::_Matrix& _dst = dynamic_cast<decx::_Matrix&>(dst);

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::sConv2_border_ignore(_src, _kernel, _dst, handle);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::sConv2_border_zero(_src, _kernel, _dst, handle);
        break;
    default:
        break;
    }
}



void decx::cuda::Conv2_fp16(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint conv_flag, const int accu_flag,
    de::DH *handle)
{
    decx::_Matrix& _src = dynamic_cast<decx::_Matrix&>(src);
    decx::_Matrix& _kernel = dynamic_cast<decx::_Matrix&>(kernel);
    decx::_Matrix& _dst = dynamic_cast<decx::_Matrix&>(dst);

    switch (conv_flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        decx::hConv2_border_ignore(_src, _kernel, _dst, handle, accu_flag);
        break;
    case decx::conv_property::de_conv_zero_compensate:
        decx::hConv2_border_zero(_src, _kernel, _dst, handle, accu_flag);
        break;
    default:
        break;
    }
}



namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint conv_flag, const int accu_flag);
    }
}



_DECX_API_
de::DH de::cuda::Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const uint conv_flag, const int accu_flag)
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
        decx::cuda::Conv2_fp32(src, kernel, dst, conv_flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP16_:
        decx::cuda::Conv2_fp16(src, kernel, dst, conv_flag, accu_flag, &handle);
        break;
    default:
        break;
    }
    return handle;
}