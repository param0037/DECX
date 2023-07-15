/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/



#ifndef _GPU_MATRIX_FILL_H_
#define _GPU_MATRIX_FILL_H_


#include "constant_fill_kernels.cuh"
#include "../../../../../classes/GPU_Matrix.h"
#include "../../../../../core/configs/config.h"


namespace de {
    namespace cuda {
        _DECX_API_ de::DH Constant_fp32(de::GPU_Matrix& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::GPU_Matrix& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::GPU_Matrix& src, const double value);
    }
}


#endif