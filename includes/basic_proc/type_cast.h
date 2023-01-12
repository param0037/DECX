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


namespace de
{
    _DECX_API_ de::Half Float2Half(const float& __x);


    _DECX_API_ float Half2Float(const de::Half& __x);
}


#endif