/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GPU_VECTOR_FILL_CUH_
#define _GPU_VECTOR_FILL_CUH_


#include "constant_fill_kernels.cuh"
#include "../../../../classes/GPU_Vector.h"


namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Constant_fp32(GPU_Vector& src, const float value);
    }
}


#endif