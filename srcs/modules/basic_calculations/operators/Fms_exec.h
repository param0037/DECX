/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FMS_EXEC_H_
#define _FMS_EXEC_H_

#include "../../core/thread_management/thread_arrange.h"
#include "../../core/thread_management/thread_pool.h"
#include "../../classes/classes_util.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/utils/fragment_arrangment.h"

// dst = C - A * B

namespace decx
{
    namespace calc {
        namespace CPUK {
            /**
            * @param A : pointer of sub-matrix A
            * @param B : pointer of sub-matrix B
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in float8
            */
            _THREAD_FUNCTION_ void 
                fms_m_fvec8_ST(const float* A, const float* B, const float* C, float* dst, size_t len);


            _THREAD_FUNCTION_ void 
                fms_m_ivec8_ST(const int* A, const int* B, const int* C, int* dst, size_t len);


            _THREAD_FUNCTION_ void 
                fms_m_dvec4_ST(const double* A, const double* B, const double* C, double* dst, size_t len);


            _THREAD_FUNCTION_ void 
                fms_c_fvec8_ST(const float* A, const float __x, const float* B, float* dst, size_t len);


            _THREAD_FUNCTION_ void
                fms_c2_fvec8_ST(const float* A, const float __x1, const float __x2, float* dst, size_t len);


            _THREAD_FUNCTION_ void 
                fms_c_ivec8_ST(const int* A, const int __x, const int* B, int* dst, size_t len);


            _THREAD_FUNCTION_ void 
                fms_c_dvec4_ST(const double* A, const double __x, const double* B, double* dst, size_t len);
        }
    }
}



#endif