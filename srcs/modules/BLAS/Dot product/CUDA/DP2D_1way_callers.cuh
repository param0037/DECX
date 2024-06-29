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


#ifndef _DP2D_1WAY_CALLERS_CUH_
#define _DP2D_1WAY_CALLERS_CUH_

#include "DP2D_1way.cuh"

namespace decx
{
    namespace blas
    {
        /**
        * @param dev_A : The device pointer where vector A is stored
        * @param dev_B : The device pointer where vector B is stored
        * @param _actual_len : The actual length of the vector, measured in element
        * @param _kp_configs : The pointer of reduction summation configuration, don't need to be initialized
        * @param S : The pointer of CUDA stream
        *
        * @return The pointer where the result being stored
        */
        template <bool _is_reduce_h>
        const void* cuda_DP2D_1way_fp32_caller_Async(decx::blas::cuda_DP2D_configs<float>* _configs, decx::cuda_stream* S);


        template <bool _is_reduce_h>
        const void* cuda_DP2D_1way_fp16_caller_Async(decx::blas::cuda_DP2D_configs<de::Half>* _configs, const uint32_t _fp16_accu, decx::cuda_stream* S);
    }
}


namespace decx
{
    namespace blas
    {
        template <bool _is_reduce_h>
        void matrix_dot_1way_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* res);


        template <bool _is_reduce_h>
        void matrix_dot_1way_fp16(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* res, const uint32_t _fp16_accu);
    }
}



#endif