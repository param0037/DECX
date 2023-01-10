/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _ADD_EXEC_H_
#define _ADD_EXEC_H_

#include "../../core/basic.h"
#include "../../core/thread_management/thread_pool.h"
#include "../../classes/classes_util.h"
#include "../../core/utils/fragment_arrangment.h"

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
                add_m_fvec8_ST(const float* A, float* B, float* dst, size_t len);


            _THREAD_FUNCTION_ void
                add_m_ivec8_ST(const __m256i* A, __m256i* B, __m256i* dst, size_t len);


            _THREAD_FUNCTION_ void
                add_m_dvec4_ST(const double* A, double* B, double* dst, size_t len);


            _THREAD_FUNCTION_ void
                add_c_fvec8_ST(const float* src, const float __x, float* dst, size_t len);


            _THREAD_FUNCTION_ void
                add_c_ivec8_ST(const __m256i* src, const int __x, __m256i* dst, size_t len);


            _THREAD_FUNCTION_ void
                add_c_dvec4_ST(const double* src, const double __x, double* dst, size_t len);
        }
    }
}


#endif