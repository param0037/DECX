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


#ifndef _EXTEND_CONSTANT_EXEC_H_
#define _EXTEND_CONSTANT_EXEC_H_


#include "../../../../core/basic.h"
#include "../../../../core/utils/intrinsics_ops.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "extend_reflect_exec_params.h"



namespace decx
{
    namespace bp {
        namespace CPUK {
            _THREAD_FUNCTION_ void _extend_constant1D_b32(const float* src, float* dst, const float val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8);


            _THREAD_FUNCTION_ void _extend_constant1D_b64(const double* src, double* dst, const double val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v4);


            _THREAD_FUNCTION_ void _extend_constant1D_b8(const uint8_t* src, uint8_t* dst, const uint8_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8);


            _THREAD_FUNCTION_ void _extend_constant1D_b16(const uint16_t* src, uint16_t* dst, const uint16_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b32(const float* src, float* dst, const float val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v8);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b64(const double* src, double* dst, const double val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v8);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b8(const uint8_t* src, uint8_t* dst, const uint8_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v16);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b16(const uint16_t* src, uint16_t* dst, const uint16_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v16);


            // vertical extension
            _THREAD_FUNCTION_ void _extend_V_constant2D_m256(float* dst, const __m256 _v_val, const uint32_t _top, const uint32_t _bottom,
                const uint Hsrc, const uint32_t Wdst);
        }
    }
}


#endif