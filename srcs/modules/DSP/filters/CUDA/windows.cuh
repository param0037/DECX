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


#ifndef _WINDOWS_CUH_
#define _WINDOWS_CUH_

// Two types of filter window (Gaussian and Triangular)
// Can be Lowpass, Bandpass or Highpass filters

#include "../../../core/basic.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../core/configs/config.h"


namespace decx
{
    namespace dsp {
        namespace GPUK {
            __global__ void
            cu_Gaussian_Window1D_cpl32(const float4* src, float4* dst, const float u, const float sigma, const size_t _proc_len, 
                const size_t real_bound);


            __global__ void
            cu_Gaussian_Window2D_cpl32_no_correlation(const float4* src, float4* dst, const float2 u, const float2 sigma, 
                const uint2 _proc_dims, const uint2 real_bound, const uint pitch);


            __global__ void
            cu_Gaussian_Window2D_cpl32(const float4* src, float4* dst, const float2 u, const float2 sigma, const uint2 _proc_dims, 
                const uint2 real_bound, const float p, const uint pitch);


            __global__ void
            cu_Triangluar_Window1D_cpl32(const float4* src, float4* dst, const long long origin, const size_t radius, 
                const size_t _proc_len, const size_t real_bound);


            __global__ void
            cu_Cone_Window2D_cpl32(const float4* src, float4* dst, const uint2 origin, const float radius, const uint2 _proc_dims,
                const uint2 real_bound, const uint pitch);
        }
    }
}



namespace de {
    namespace dsp {
        namespace cuda {
            _DECX_API_ de::DH Gaussian_Window1D(de::GPU_Vector& src, de::GPU_Vector& dst, const float u, const float sigma);

            /*
            * @param u : The mean value of the Gaussian filter
            * @param sigma : The deviation of the Gaussian filter
            * @param p : The corrolation factor of the Gaussian filter
            */
            _DECX_API_ de::DH Gaussian_Window2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p);


            _DECX_API_ de::DH Triangular_Window1D(de::GPU_Vector& src, de::GPU_Vector& dst, const long long origin, const size_t radius);


            _DECX_API_ de::DH Cone_Window2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D origin, const float radius);
        }
    }
}


#endif