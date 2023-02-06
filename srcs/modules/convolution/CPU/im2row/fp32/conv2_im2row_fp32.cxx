/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "conv2_im2row_fp32.h"
#include "../../../conv_utils.h"


namespace de {
    namespace cpu {
        _DECX_API_ de::DH Conv2D(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const de::Point2D strideXY, const int conv_flag);
    }
}


_DECX_API_ de::DH de::cpu::Conv2D(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const de::Point2D strideXY, const int conv_flag)
{
    de::DH handle;
    if (!decx::cpI.is_init) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    decx::_TensorArray* _kernel = dynamic_cast<decx::_TensorArray*>(&kernel);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    const uint2 _strideXY = make_uint2(strideXY.x, strideXY.y);
    
    switch (conv_flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_im2row_fp32_NB(_src, _kernel, _dst, &handle);
        }
        else {
            decx::conv_I2R::conv2_im2row_fp32_NB_stride(_src, _kernel, _dst, _strideXY, &handle);
        }
        break;

    case decx::conv_property::de_conv_zero_compensate:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_im2row_fp32_BC(_src, _kernel, _dst, &handle);
        }
        else {
            decx::conv_I2R::conv2_im2row_fp32_BC_stride(_src, _kernel, _dst, _strideXY, &handle);
        }
        break;
    default:
        break;
    }
    
    decx::err::Success(&handle);
    return handle;
}