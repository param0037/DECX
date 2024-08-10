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


#ifndef _ROTATION_H_
#define _ROTATION_H_


#include "../../../../../common/Classes/Tensor.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"



namespace decx {
    namespace fields {
        namespace CPUK {
            static inline _THREAD_CALL_ __m128
                rotcalc_internal_shuffle(decx::utils::simd::xmm128_reg* _px, decx::utils::simd::xmm128_reg* _py,
                    decx::utils::simd::xmm128_reg* _pz);

            /**
            * @param _proc_dims : ~.x : x_axis dims; ~.y : y_axis dims; ~.y : y_axis dims;
            * @param dp_x_wp : In float;
            * @param d_pitch : In float;
            */
            _THREAD_FUNCTION_ void
                rotation_field3D_fp32(const float* src, float* dst, const uint3 _proc_dims, const uint dp_x_wp, const uint d_pitch);
        }
    }
}


#endif