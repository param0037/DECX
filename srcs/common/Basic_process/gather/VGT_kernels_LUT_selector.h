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

#ifndef _VGT_KERNELS_LUT_SELECTOR_H_
#define _VGT_KERNELS_LUT_SELECTOR_H_

#include "interpolate_types.h"

namespace decx
{
    template<typename _type_in, typename _type_out>
#ifdef _DECX_CUDA_PARTS_
    static uint2 VGT2D_kernel_selector()
#else
    static uint2 VGT2D_kernel_selector(const de::Interpolate_Types intp_type)
#endif
    {
        uint2 res = {0, 0};

#if __cplusplus >= 201103L
        if (std::is_same<_type_in, float>::value)           res.x = 0;
        else if (std::is_same<_type_in, uint8_t>::value)    res.x = 1;
        else if (std::is_same<_type_in, uchar4>::value)     res.x = 2;

        if (std::is_same<_type_out, float>::value)          res.y = 0;
        else if (std::is_same<_type_out, uint8_t>::value)   res.y = 1;
        else if (std::is_same<_type_out, uchar4>::value)    res.y = 2;
        
#endif
#if __cplusplus >= 201703L
        if constexpr (std::is_same_v<_type_in, float>)          res.x = 0;
        else if constexpr (std::is_same_v<_type_in, uint8_t>)   res.x = 1;
        else if constexpr (std::is_same_v<_type_in, uchar4>)    res.x = 2;

        if constexpr (std::is_same_v<_type_out, float>)         res.y = 0;
        else if constexpr (std::is_same_v<_type_out, uint8_t>)  res.y = 1;
        else if constexpr (std::is_same_v<_type_out, uchar4>)   res.y = 2;
        
#endif
#ifdef _DECX_CPU_PARTS_
        res.y = res.y * 2 + (uint32_t)intp_type;
#endif
        return res;
    }
}

#endif
