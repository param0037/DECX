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


#ifndef _LOW_PASS_H_
#define _LOW_PASS_H_

#include "../../../../common/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../../common/SIMD/intrinsics_ops.h"
#include "../../../../common/Classes/Vector.h"
#include "../../../../common/Classes/Matrix.h"
#include "../../../../common/FMGR/fragment_arrangment.h"


namespace decx {
    namespace dsp {
        namespace CPUK {
            _THREAD_FUNCTION_ void
            ideal_LP1D_cpl32_ST(const double* src, double* dst, const size_t cutoff_freq, const size_t _proc_len, 
                const size_t real_bound, const size_t global_dex_offset);

            /*
            * @param _proc_dims : ~.x -> width of the processed area; ~.y -> height of the processed area;
            * @param real_bound : ~.x -> x-axis of the processed area; ~.y -> y-axis of the processed area;
            * @param pitch : pitch of the processed area, in float4 (vec2 of datatype of de::CPf);
            * @param global_dex_offset : ~.x -> offset on x-axis; ~.y -> offset on y-axis;
            */
            _THREAD_FUNCTION_ void
            ideal_LP2D_cpl32_ST(const double* src, double* dst, const uint2 _proc_dims, const uint2 real_bound, 
                const uint2 cutoff_freq, const uint pitch, const uint2 global_dex_offset);


            
            _THREAD_FUNCTION_ void
            ButterWorth_Window2D_cpl32(const double* src, double* dst, const int order, const float cutoff_freq, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);

        }
    }
}



namespace de {
    namespace dsp {
        namespace cpu {
            _DECX_API_ de::DH LowPass1D_Ideal(de::Vector& src, de::Vector& dst, const size_t cutoff_frequency);


            _DECX_API_ de::DH LowPass2D_Ideal(de::Matrix& src, de::Matrix& dst, const de::Point2D cutoff_frequency);


            _DECX_API_ de::DH ButterWorth_LP2D(de::Matrix& src, de::Matrix& dst, const float cutoff_freq, const int order);
        }
    }
}


#endif