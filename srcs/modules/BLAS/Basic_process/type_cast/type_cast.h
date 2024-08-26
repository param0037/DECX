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


#ifndef _MATRIX_TYPE_CAST_H_
#define _MATRIX_TYPE_CAST_H_


#include "../../../../common/Classes/Matrix.h"
#include "../../../../common/Classes/MatrixArray.h"
#include "../../../../common/Classes/Vector.h"
#include "../../../../common/Classes/Tensor.h"
#include "../../../../common/Basic_process/type_cast/CPU/typecast_exec_x86_64.h"


namespace decx
{
namespace type_cast
{
    namespace cpu {
        _DECX_API_ void _type_cast1D_organiser(void* src, void* dst, const uint64_t proc_len, const int32_t cvt_method, de::DH* handle);


        _DECX_API_ void _type_cast2D_organiser(void* src, void* dst, const uint2 proc_dims, const uint32_t Wsrc,
            const uint32_t Wdst, const int cvt_method, de::DH* handle);
    }
}
}



namespace de {
    namespace cpu 
    {
        _DECX_API_ void TypeCast(de::Vector& src, de::Vector& dst, const int cvt_method);


        _DECX_API_ void TypeCast(de::Matrix& src, de::Matrix& dst, const int cvt_method);


        _DECX_API_ void TypeCast(de::MatrixArray& src, de::MatrixArray& dst, const int cvt_method);


        _DECX_API_ void TypeCast(de::Tensor& src, de::Tensor& dst, const int cvt_method);
    }
}


#endif