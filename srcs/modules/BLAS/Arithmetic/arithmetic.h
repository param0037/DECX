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

#ifndef _ARITHMETIC_H_
#define _ARITHMETIC_H_

#include "../../../common/Classes/Vector.h"
#include "../../../common/Classes/Matrix.h"
#include "../../../common/Classes/Number.h"


namespace de
{
    enum DecxArithmetic{
        ADD = 0x00,
        SUB = 0x01,
        MUL = 0x02,
        DIV = 0x03,
        MIN = 0x04,
        MAX = 0x05,
        COS = 0x06,
        SIN = 0x07,

        OP_INV = 0x20
    };
}


namespace de
{
namespace blas{
namespace cpu{
    _DECX_API_ void Arithmetic(de::InputVector A, de::InputVector B, de::OutputVector dst, const int32_t arith_flag);
    _DECX_API_ void Arithmetic(de::InputVector src, de::InputNumber constant, de::OutputVector dst, const int32_t arith_flag);
}

namespace cpu{
    _DECX_API_ void Arithmetic(de::InputMatrix A, de::InputMatrix B, de::OutputMatrix dst, const int32_t arith_flag);
}
}

}


#endif