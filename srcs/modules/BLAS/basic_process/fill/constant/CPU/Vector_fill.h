/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _VECTOR_FILL_H_
#define _VECTOR_FILL_H_


#include "fill_constant_exec.h"
#include "../../../../../classes/Vector.h"
#include "../../../../../core/configs/config.h"


namespace de {
    namespace cpu {
        _DECX_API_ de::DH Constant_fp32(de::Vector& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::Vector& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::Vector& src, const double value);
    }
}


#endif