/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_UTILS_H_
#define _GEMM_UTILS_H_

#include "../../core/basic.h"
#include "../../classes/Matrix.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/resources_manager/decx_resource.h"


#define GEMM_BlockDim 16

namespace de
{
    enum GEMM_properties
    {
        HALF_GEMM_DIRECT    = 0,
        HALF_GEMM_ACCURATE  = 1
    };
}


#ifdef _DECX_BLAS_CPU_


#endif

#endif