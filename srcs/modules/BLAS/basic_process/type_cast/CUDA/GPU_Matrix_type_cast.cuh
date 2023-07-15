/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GPU_MATRIX_TYPE_CAST_CUH_
#define _GPU_MATRIX_TYPE_CAST_CUH_


#include "_mm128_fp32_fp64.cuh"
#include "_mm128_fp32_int32.cuh"
#include "../../../../classes/GPU_Matrix.h"


namespace de {
    namespace cuda {
        _DECX_API_ de::DH TypeCast(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int cvt_method);
    }
}


#endif