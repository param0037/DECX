/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "float_half_convert.h"


de::Half de::Float2Half(const float& __x)
{
    __half tmp = __float2half(__x);
    return *((de::Half*)&tmp);
}


float de::Half2Float(const de::Half& __x)
{
    float tmp = __half2float(*((__half*)&__x));
    return tmp;
}