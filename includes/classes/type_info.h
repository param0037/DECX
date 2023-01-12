/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _TYPE_INFO_H_
#define _TYPE_INFO_H_

#include"class_utils.h"

namespace de
{
    enum _DATA_TYPES_FLAGS_ {
        _VOID_ = 0,
        _INT32_ = 1,
        _FP32_ = 2,
        _FP64_ = 3,
        _FP16_ = 4,
        _COMPLEX_F32_ = 5,
        _UINT8_ = 6,
        _UCHAR3_ = 7,
        _UCHAR4_ = 8
    };
}


#endif