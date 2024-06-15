
#ifndef _GEMM_64B_CALLER_H_
#define _GEMM_64B_CALLER_H_

#include "../../../../classes/Matrix.h"
#include "GEMM_Matrix_B_arrange_64b.h"
#include "../../../../classes/classes_util.h"
#include "GEMM_organizer_64b.h"
#include "GEMM_ABC_organizer_64b.h"
#include "../../../basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../GEMM_utils.h"



namespace decx
{
    namespace cpu
    {
        template <bool _is_cpl>
        void GEMM_64b(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle);

        template <bool _is_cpl>
        void GEMM_64b_ABC(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH* handle);
    }
}


#endif
