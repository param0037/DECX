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


#ifndef _MATRIX_REDUCE_H_
#define _MATRIX_REDUCE_H_

#include "../../../classes/Matrix.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../core/configs/config.h"
#include "../../../classes/classes_util.h"
#include "../../../classes/DecxNumber.h"
#ifdef _DECX_CUDA_PARTS_
#include "../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../classes/GPU_Matrix.h"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#endif


namespace de
{
    enum REDUCE_METHOD
    {
        _REDUCE2D_FULL_ = 0,
        _REDUCE2D_H_ = 1,
        _REDUCE2D_V_ = 2,
    };


    namespace cuda 
    {
        // 1-way
        _DECX_API_ de::DH Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);

        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);

        // Full
        _DECX_API_ de::DH Sum(de::Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::Matrix& src, de::DecxNumber& res);
        _DECX_API_ de::DH Min(de::Matrix& src, de::DecxNumber& res);

        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::DecxNumber& res);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::DecxNumber& res);
    }
}


#endif