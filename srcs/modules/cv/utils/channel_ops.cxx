/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "channel_ops.h"


de::DH de::vis::merge_channel(de::Matrix& src, de::Matrix& dst, const int flag)
{
    de::DH handle;
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (flag)
    {
    case de::vis::ImgChannelMergeType::BGR_to_Gray:
        decx::vis::_channel_ops_general_caller(decx::vis::_BGR2Gray_ST_UC2UC,
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->width, _src->height),
            _src->pitch, _dst->pitch);
        break;

    case de::vis::ImgChannelMergeType::RGB_mean:
        decx::vis::_channel_ops_general_caller(decx::vis::_BGR2Mean_ST_UC2UC,
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height), 
            _src->pitch, _dst->pitch);
        break;

    case de::vis::ImgChannelMergeType::Preserve_B:
        decx::vis::_channel_ops_general_caller(decx::vis::_Preserve_B_ST_UC2UC,
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height), 
            _src->pitch, _dst->pitch);
        break;

    case de::vis::ImgChannelMergeType::Preserve_G:
        decx::vis::_channel_ops_general_caller(decx::vis::_Preserve_G_ST_UC2UC,
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height), 
            _src->pitch, _dst->pitch);
        break;

    case de::vis::ImgChannelMergeType::Preserve_R:
        decx::vis::_channel_ops_general_caller(decx::vis::_Preserve_R_ST_UC2UC,
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height), 
            _src->pitch, _dst->pitch);
        break;

    case de::vis::ImgChannelMergeType::Preserve_Alpha:
        decx::vis::_channel_ops_general_caller(decx::vis::_Preserve_A_ST_UC2UC,
            reinterpret_cast<float*>(_src->Mat.ptr), reinterpret_cast<float*>(_dst->Mat.ptr), make_int2(_src->pitch, _src->height), 
            _src->pitch, _dst->pitch);
        break;

    default:
        break;
    }


    return handle;
}