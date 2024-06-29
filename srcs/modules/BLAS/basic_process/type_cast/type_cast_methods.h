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


#ifndef _TYPE_CAST_METHOD_H_
#define _TYPE_CAST_METHOD_H_


namespace decx {
    namespace type_cast 
    {
        enum TypeCast_Method {
            CVT_INT32_UINT8 = 0,

            CVT_UINT8_CLAMP_TO_ZERO = 1,
            CVT_INT32_UINT8_TRUNCATE = 2,

            CVT_FP32_FP64 = 4,

            CVT_UINT8_CYCLIC = 5,

            CVT_FP64_FP32 = 6,
            CVT_INT32_FP32 = 7,
            CVT_FP32_INT32 = 8,

            CVT_UINT8_SATURATED = 9,

            CVT_UINT8_INT32 = 10,
            
            CVT_FP32_UINT8 = 16,
            CVT_UINT8_FP32 = 17
        };
    }
}


#endif