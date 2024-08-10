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


#ifndef _WINDOWS_H_
#define _WINDOWS_H_


#include "../../../../common/basic.h"
#include "../../../../common/Classes/Matrix.h"
#include "../../../../common/Classes/Vector.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../../common/SIMD/intrinsics_ops.h"
#include "../../../../common/FMGR/fragment_arrangment.h"


namespace decx {
    namespace dsp {
        namespace CPUK 
        {
            /*
            * @param _proc_len : Length of processed area, in __m256 (vec4) (de::CPf x4)
            * @param real_bound : Real length of the array, in de::CPf (its own element)
            * @param global_dex_offset : Offset of global dex, in its own element (de::CPf)
            */
            _THREAD_FUNCTION_ void
            Gaussian_Window1D_cpl32(const double* src, double* dst, const float u, const float sigma, const size_t _proc_len,
                const size_t real_bound, const size_t global_dex_offset);


            _THREAD_FUNCTION_ void
            Gaussian_Window2D_cpl32_no_corrolation(const double* src, double* dst, const float2 u, const float2 sigma, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);


            _THREAD_FUNCTION_ void
            Gaussian_Window2D_cpl32(const double* src, double* dst, const float2 u, const float2 sigma, const float p, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);



            _THREAD_FUNCTION_ void
            Cone_Window2D_cpl32(const double* src, double* dst, const uint2 origin, const float radius, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);



            /**
            * @param _proc_len : Length of processed area, in __m256 (vec4) (de::CPf x4)
            * @param real_bound : Real length of the array, in de::CPf (its own element)
            * @param global_dex_offset : Offset of global dex, in its own element (de::CPf)
            */
            _THREAD_FUNCTION_ void
            Triangular_Window1D_cpl32(const double* src, double* dst, const long long center, const size_t radius, const size_t _proc_len,
                const size_t real_bound, const size_t global_dex_offset);
        }
    }
}



namespace de {
    namespace dsp {
        namespace cpu {
            _DECX_API_ de::DH Gaussian_Window1D(de::Vector& src, de::Vector& dst, const float u, const float sigma);


            _DECX_API_ de::DH Triangular_Window1D(de::Vector& src, de::Vector& dst, const long long center, size_t radius);


            _DECX_API_ de::DH Gaussian_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p);


            _DECX_API_ de::DH Cone_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D origin, const float radius);
        }
    }
}


#endif