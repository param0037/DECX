/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _VECTOR_TYPE_CAST_H_
#define _VECTOR_TYPE_CAST_H_


#include "../../../classes/Vector.h"
#include "_mm256_fp32_fp64.h"
#include "_mm256_fp32_int32.h"
#include "_mm256_uint8_int32.h"


namespace de {
    namespace cpu {
        _DECX_API_ de::DH TypeCast(de::Vector& src, de::Vector& dst, const int cvt_method);
    }
}


#endif