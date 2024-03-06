/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "channel_ops.h"


#define _IMG_CHANNEL_OP_UC42UC_CALL_(op)            \
    decx::vis::_channel_ops_UC42UC_caller(op,       \
    reinterpret_cast<const float*>(_src->Mat.ptr),  \
    reinterpret_cast<float*>(_dst->Mat.ptr),        \
    make_int2(_src->Width(), _src->Height()),       \
    _src->Pitch(),                                  \
    _dst->Pitch());                                 \


#define _IMG_CHANNEL_OP_UC42UC4_CALL_(op)           \
    decx::vis::_channel_ops_UC42UC4_caller(op,      \
    reinterpret_cast<const float*>(_src->Mat.ptr),  \
    reinterpret_cast<float*>(_dst->Mat.ptr),        \
    make_int2(_src->Width(), _src->Height()),       \
    _src->Pitch(),                                  \
    _dst->Pitch());                                 \



_DECX_API_
de::DH de::vis::ColorTransform(de::Matrix& src, de::Matrix& dst, const de::vis::color_transform_types flag)
{
    de::DH handle;
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    switch (flag)
    {
    case de::vis::color_transform_types::RGB_to_Gray:
        _IMG_CHANNEL_OP_UC42UC_CALL_(decx::vis::CPUK::_BGR2Gray_UC42UC);
        break;

    case de::vis::color_transform_types::RGB_mean:
        _IMG_CHANNEL_OP_UC42UC_CALL_(decx::vis::CPUK::_BGR2Mean_UC42UC);
        break;

    case de::vis::color_transform_types::Preserve_B:
        _IMG_CHANNEL_OP_UC42UC_CALL_(decx::vis::CPUK::_Preserve_B_UC42UC);
        break;

    case de::vis::color_transform_types::Preserve_G:
        _IMG_CHANNEL_OP_UC42UC_CALL_(decx::vis::CPUK::_Preserve_G_UC42UC);
        break;

    case de::vis::color_transform_types::Preserve_R:
        _IMG_CHANNEL_OP_UC42UC_CALL_(decx::vis::CPUK::_Preserve_R_UC42UC);
        break;

    case de::vis::color_transform_types::Preserve_Alpha:
        _IMG_CHANNEL_OP_UC42UC_CALL_(decx::vis::CPUK::_Preserve_A_UC42UC);
        break;

    case de::vis::color_transform_types::RGB_to_YUV:
        _IMG_CHANNEL_OP_UC42UC4_CALL_(decx::vis::CPUK::_RGB2YUV_UC42UC4);
        _dst->set_data_format(de::_DATA_FORMATS_::_COLOR_YUV_);
        break;

    case de::vis::color_transform_types::YUV_to_RGB:
        _IMG_CHANNEL_OP_UC42UC4_CALL_(decx::vis::CPUK::_YUV2RGB_UC42UC4);
        _dst->set_data_format(de::_DATA_FORMATS_::_COLOR_RGB_);
        break;

    case de::vis::color_transform_types::RGB_to_BGR:
        _IMG_CHANNEL_OP_UC42UC4_CALL_(decx::vis::CPUK::_RGB2BGR_UC42UC4);
        _dst->set_data_format(de::_DATA_FORMATS_::_COLOR_BGR_);
        break;

    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag,
            MEANINGLESS_FLAG);
        break;
    }

    return handle;
}


