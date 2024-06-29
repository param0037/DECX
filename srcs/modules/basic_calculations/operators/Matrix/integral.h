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


#ifndef _INTEGRAL_H_
#define _INTEGRAL_H_


#include "../../../classes/Matrix.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../core/configs/config.h"


namespace decx
{
    namespace calc {
        namespace CPUK {
            /**
            * This function integral each line of the matrix
            * @param proc_dims : ~.x -> width of process area, in float; ~.y -> height of process area, in float
            * @param pitchsrc : the pitch of src, in float
            * @param pitchdst : the pitch of dst, in float
            */
            _THREAD_FUNCTION_ void 
            _integral_H_fp32_2D_ST(const float* src, float* dst, const uint2 proc_dims, const uint pitchsrc, const uint pitchdst);


            /**
            * This function integral each line of the matrix
            * @param proc_dims : ~.x -> width of process area, in __m256; ~.y -> height of process area, in float
            * @param pitch : the pitch of dst, in float
            */
            _THREAD_FUNCTION_ void 
            _integral_V_fp32_2D_ST(float* src, const uint2 proc_dims, const uint pitch);



            /**
            * This function integral each line of the matrix
            * @param proc_dims : ~.x -> width of process area, in __m256; ~.y -> height of process area, in float
            * @param pitch : the pitch of dst, in float
            */
            _THREAD_FUNCTION_ void 
            _integral_V_uint8_2D_ST(float* src, const uint2 proc_dims, const uint pitch);


            /**
            * This function integral each line of the matrix
            * @param proc_dims : ~.x -> width of process area, in __m256; ~.y -> height of process area, in float
            * @param pitch : the pitch of dst, in float
            */
            _THREAD_FUNCTION_ void 
            _integral_V_uint8_f32_2D_ST(float* src, const uint2 proc_dims, const uint pitch);



            /**
            * This function integral each line of the matrix
            * @param proc_dims : ~.x -> width of process area, in float; ~.y -> height of process area, in float
            * @param pitchsrc : the pitch of src, in float
            * @param pitchdst : the pitch of dst, in float
            */
            _THREAD_FUNCTION_ void 
            _integral_H_uint8_2D_ST(const uint64_t* src, float* dst, const uint2 proc_dims, const uint pitchsrc, const uint pitchdst);

        }

        _DECX_API_ void _integral_caller2D_fp32(const float* src, float* dst, const uint2 proc_dims,
            const uint pitchsrc, const uint pitchdst);


        _DECX_API_ void _integral_caller2D_uint8(const uint8_t* src, int32_t* dst, const uint2 proc_dims,
            const uint pitchsrc, const uint pitchdst);


        _DECX_API_ void _integral_caller2D_uint8_f32(const uint8_t* src, float* dst, const uint2 proc_dims,
            const uint pitchsrc, const uint pitchdst);
    }
}


namespace de
{
    namespace calc {
        namespace cpu {
            _DECX_API_ de::DH Integral(de::Matrix& src, de::Matrix& dst);
        }
    }
}


#endif