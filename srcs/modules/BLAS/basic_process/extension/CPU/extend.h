/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _BORDER_H_
#define _BORDER_H_


#include "extend_reflect.h"
#include "extend_constant.h"
#include "../extend_flags.h"


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Extend(de::Vector& src, de::Vector& dst, const uint32_t left, const uint32_t right,
            const int border_type, void* val);



        _DECX_API_ de::DH Extend(de::Matrix& src, de::Matrix& dst, const uint32_t left, const uint32_t right,
            const uint32_t top, const uint32_t bottom, const int border_type, void* val);
    }
}


#endif