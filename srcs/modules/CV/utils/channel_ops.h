/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CV_CLS_MFUNCS_H_
#define _CV_CLS_MFUNCS_H_

#include "../../core/basic.h"
#include "../../classes/Matrix.h"
#include "../../core/memory_management/MemBlock.h"
#include "../../handles/decx_handles.h"
#ifdef _DECX_CPU_PARTS_
#include "../utils/cvt_colors.h"
#endif



namespace de
{
    namespace vis
    {
        enum color_transform_types {
            RGB_to_Gray = 0,
            Preserve_B = 1,
            Preserve_G = 2,
            Preserve_R = 3,
            Preserve_Alpha = 4,
            RGB_mean = 5,
            RGB_to_YUV = 6,
            YUV_to_RGB = 7,
            RGB_to_BGR = 8
        };

        _DECX_API_ de::DH ColorTransform(de::Matrix& src, de::Matrix& dst, const de::vis::color_transform_types flag);
    }
}

#endif