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


#ifndef _MATRIX_RECT_H_
#define _MATRIX_RECT_H_

#include "../../../classes/Matrix.h"
#include "rect_copy2D_exec.h"


namespace de
{
    namespace cpu
    {
        template <typename T>
        _DECX_API_ de::DH Rect(de::Matrix<T>& src, de::Matrix<T>& dst, const de::Point2D start, const de::Point2D end);
    }
}

template <typename T>
de::DH de::cpu::Rect(de::Matrix<T>& src, de::Matrix<T>& dst, const de::Point2D start, const de::Point2D end)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix<T>* _src = dynamic_cast<decx::_Matrix<T>*>(&src);
    decx::_Matrix<T>* _dst = dynamic_cast<decx::_Matrix<T>*>(&dst);

    // ~.x -> width; ~.y -> height
    uint2 cpy_dim = make_uint2(decx::utils::clamp_max<uint>(end.y, _src->width) - start.y,
                               decx::utils::clamp_max<uint>(end.x, _src->height) - start.x);
    
    decx::_cpy2D_anybit_caller<T>(DECX_PTR_SHF_XY_SAME_TYPE<T>(_src->Mat.ptr, start.x, start.y, _src->pitch), 
        _dst->Mat.ptr, cpy_dim);

    decx::err::Success(&handle);
    return handle;
}


#endif