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


#ifndef _LOG_EXEC_H_
#define _LOG_EXEC_H_


#include "../../core/basic.h"
#include "../../core/thread_management/thread_pool.h"


namespace decx
{
    namespace calc {
        namespace CPUK {
            /**
            * dst(i) = log10(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void log10_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void log10_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = log2(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void log2_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void log2_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = sin(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void sin_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void sin_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = cos(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void cos_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void cos_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = tan(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void tan_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void tan_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void exp_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void exp_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void acos_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void acos_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void asin_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void asin_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void atan_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void atan_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void sinh_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void sinh_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void cosh_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void cosh_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void tanh_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void tanh_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = exp(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void abs_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void abs_dvec4_ST(const double* src, double* dst, uint64_t len);

            /**
            * dst(i) = sqrt(src(i))
            * @param src : pointer of sub-matrix src
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void sqrt_fvec8_ST(const float* src, float* dst, uint64_t len);
            _THREAD_FUNCTION_ void sqrt_dvec4_ST(const double* src, double* dst, uint64_t len);
        }
    }
}



#endif