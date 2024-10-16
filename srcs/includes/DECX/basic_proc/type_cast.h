/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TYPE_CAST_H_
#define _TYPE_CAST_H_


#include "../basic.h"
#include "../classes/class_utils.h"


namespace de {

    enum TypeCast_Method {
        CVT_INT32_UINT8             = 0,

        CVT_UINT8_CLAMP_TO_ZERO     = 1,
        CVT_INT32_UINT8_TRUNCATE    = 2,

        CVT_FP32_FP64               = 4,

        CVT_UINT8_CYCLIC            = 5,

        CVT_FP64_FP32               = 6,
        CVT_INT32_FP32              = 7,
        CVT_FP32_INT32              = 8,

        CVT_UINT8_SATURATED         = 9,

        CVT_UINT8_INT32             = 10,

        CVT_FP32_UINT8              = 16,
        CVT_UINT8_FP32              = 17
    };

    namespace cpu 
    {
        _DECX_API_ void TypeCast(de::InputVector src, de::OutputVector& dst, const int32_t cvt_method);


        _DECX_API_ void TypeCast(de::InputMatrix src, de::OutputMatrix& dst, const int32_t cvt_method);
    }
}


#endif
