/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CP_OPS_EXEC_H_
#define _CP_OPS_EXEC_H_

#include "../../core/basic.h"
#include "../../core/thread_management/thread_pool.h"
#include "../../classes/classes_util.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../signal/CPU_cpf32_avx.h"


namespace decx
{
    namespace calc {
        namespace CPUK {
            /**
            * @param A : pointer of sub-matrix A
            * @param B : pointer of sub-matrix B
            * @param dst : pointer of sub-matrix dst
            * @param len : regard the data space as a 1D array, the length is in double8
            */
            _THREAD_FUNCTION_ void
                cp_add_m_fvec4_ST(const double* A, double* B, double* dst, size_t len);
            

            _THREAD_FUNCTION_ void
                cp_sub_m_fvec4_ST(const double* A, double* B, double* dst, size_t len);



            _THREAD_FUNCTION_ void
                cp_mul_m_fvec4_ST(const double* A, double* B, double* dst, size_t len);



            _THREAD_FUNCTION_ void
                cp_add_c_fvec4_ST(const double* src, const double __x, double* dst, size_t len);


            _THREAD_FUNCTION_ void
                cp_sub_c_fvec4_ST(const double* src, const double __x, double* dst, size_t len);


            _THREAD_FUNCTION_ void
                cp_mul_c_fvec4_ST(const double* src, const double __x, double* dst, size_t len);
        }
    }
}


#endif