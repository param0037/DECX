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


#ifndef _ROTATE_FP32_H_
#define _ROTATE_FP32_H_


#include "../../basic_calculations/operators/cp_ops_exec.h"
#include "../../classes/classes_util.h"
#include "../../classes/Vector.h"
#include "../../classes/Matrix.h"


namespace decx {
    namespace dsp {
        /*
        * @param src : Input pointer
        * @param angle : The rotating angle
        * @param src : Output pointer
        * @param _proc_len : In vec4 (de::CPf x4)
        */
        void complex_rotate_fp32_caller(const double* src, const float angle, double* dst, const size_t _proc_len);


        namespace CPU {
            _DECX_API_ de::DH Complex_Rotate(de::Matrix& src, const float angle, de::Matrix& dst);


            _DECX_API_ de::DH Complex_Rotate(de::Vector& src, const float angle, de::Vector& dst);
        }
    }
}





#endif