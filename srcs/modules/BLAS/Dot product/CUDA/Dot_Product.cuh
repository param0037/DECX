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


#ifndef _DP1D_CUH_
#define _DP1D_CUH_


#include <Classes/Vector.h>
#include <Classes/Matrix.h>
#include <Classes/GPU_Vector.h>
#include <Classes/Number.h>
#include "../../Basic_process/Matrix_reduce.h"


namespace de
{
namespace blas {
    namespace cuda
    {
        _DECX_API_ void Dot_product(de::Vector& A, de::Vector& B, de::Number& res, const uint32_t _fp16_accu);

        _DECX_API_ void Dot_product(de::GPU_Vector& A, de::GPU_Vector& B, de::Number& res, const uint32_t _fp16_accu);

        _DECX_API_ void Dot_product(de::Matrix& A, de::Matrix& B, de::Vector& dst, const de::REDUCE_METHOD _rd_method, const uint32_t _fp16_accu);
    }
}
}



#endif