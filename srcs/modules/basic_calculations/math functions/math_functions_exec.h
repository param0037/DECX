/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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