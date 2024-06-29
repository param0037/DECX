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