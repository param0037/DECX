/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _TYPE_CAST_METHOD_H_
#define _TYPE_CAST_METHOD_H_


namespace decx {
    namespace type_cast 
    {
        enum TypeCast_Method {
            CVT_FP32_FP64 = 0,
            CVT_FP64_FP32 = 1,
            CVT_FP32_INT32 = 2,
            CVT_INT32_FP32 = 3,
            CVT_INT32_UINT8 = 4,
            CVT_UINT8_INT32 = 5,

            CVT_INT32_UINT8_TRUNCATE = 6,
            CVT_INT32_UINT8_CYCLIC = 7,
            CVT_INT32_UINT8_SATURATED = 8,
            CVT_INT32_UINT8_CLAMP_TO_ZERO = 10
        };
    }
}


#endif
