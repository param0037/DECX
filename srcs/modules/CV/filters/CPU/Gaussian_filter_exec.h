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


#ifndef _GAUSSIAN_FILTER_EXEC_H_
#define _GAUSSIAN_FILTER_EXEC_H_


#include "../../../core/basic.h"
#include "../../../DSP/convolution/CPU/sliding_window/uint8/conv2_uint8_K_loop_core.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../core/allocators.h"


namespace decx
{
    namespace vis {
        namespace CPUK {
            _THREAD_FUNCTION_
                void _gaussian_H_uint8_ST(const double* src, const float* kernel, double* dst, const uint2 proc_dim, const uint32_t Wker,
                    const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }

        void _gaussian_H_uint8_caller(const double* src, const float* kernel, float* dst, const uint2 proc_dim, const uint32_t Wker,
            const uint Wsrc, const uint Wdst, const ushort reg_WL, decx::utils::_thr_1D* t1D, const uint _loop);


        void _gaussian_H_uchar4_caller(const float* src, const float* kernel, float* dst, const uint2 proc_dim, const uint32_t Wker,
            const uint Wsrc, const uint Wdst, const ushort reg_WL, decx::utils::_thr_1D* t1D, const uint _loop);


        void _gaussian_V_uint8_caller(const float* src, const float* kernel, double* dst, const uint2 proc_dim, const uint32_t Hker,
            const uint Wsrc, const uint Wdst, decx::utils::_thr_1D* t1D);
    }
}



#endif