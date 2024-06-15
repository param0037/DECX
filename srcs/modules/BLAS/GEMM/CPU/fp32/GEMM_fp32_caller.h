/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_FP32_CALLER_H_
#define _GEMM_FP32_CALLER_H_

#include "../../../../classes/Matrix.h"
#include "GEMM_Matrix_B_arrange_fp32.h"
#include "../../../../classes/classes_util.h"
#include "GEMM_organizer_fp32.h"
#include "GEMM_ABC_organizer_fp32.h"
#include "../../../basic_process/rect_and_cube/CPU/rect_copy2D_exec.h"
#include "../../GEMM_utils.h"


namespace decx
{
    namespace cpu
    {
        void GEMM_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, de::DH* handle);


        void GEMM_fp32_ABC(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* C, decx::_Matrix* dst, de::DH* handle);
    }
}



#endif