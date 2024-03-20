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
#include "../../../../DSP/convolution/conv_utils.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace de {
    namespace cpu {
        _DECX_API_ de::DH Conv2D(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const de::Point2D strideXY, const int conv_flag);
    }
}


_DECX_API_ de::DH de::cpu::Conv2D(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const de::Point2D strideXY, const int conv_flag)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Tensor* _src = dynamic_cast<decx::_Tensor*>(&src);
    decx::_TensorArray* _kernel = dynamic_cast<decx::_TensorArray*>(&kernel);
    decx::_Tensor* _dst = dynamic_cast<decx::_Tensor*>(&dst);

    const uint2 _strideXY = make_uint2(strideXY.x, strideXY.y);
    
    switch (conv_flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        if (strideXY.x < 2 && strideXY.y < 2) {
            decx::conv_I2R::conv2_im2row_fp32_NB(_src, _kernel, _dst, &handle);
        }
        else {
            decx::conv_I2R::conv2_im2row_fp32_NB_stride(_src, _kernel, _dst, _strideXY, &handle);
        }
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
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