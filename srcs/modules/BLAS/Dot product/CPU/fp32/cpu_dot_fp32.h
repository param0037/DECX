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


#ifndef _CPU_DOT_FP32_H_
#define _CPU_DOT_FP32_H_

#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../../common/Classes/Vector.h"
#include "../../../../../common/Classes/Matrix.h"
#include "../../../../../common/Classes/Tensor.h"
#include "../../../../../common/FMGR/fragment_arrangment.h"
#include "../../../../../common/Classes/classes_util.h"
#include "../../../../../common/SIMD/intrinsics_ops.h"


namespace decx
{
    namespace dot {
        namespace CPUK {
            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            _THREAD_FUNCTION_ void
                _dot_vec8_fp32(const float* A, const float* B, const size_t len, float* res_vec);
        }

        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        void _dot_fp32_1D_caller(const float* A, const float* B, const size_t len, float* res_vec);
    }
}



namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Dot(de::Vector& A, de::Vector& B, float* res);


        _DECX_API_ de::DH Dot(de::Matrix& A, de::Matrix& B, float* res);


        _DECX_API_ de::DH Dot(de::Tensor& A, de::Tensor& B, float* res);
    }
}



#endif