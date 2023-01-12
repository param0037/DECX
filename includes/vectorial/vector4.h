/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _VECTOR4_H_
#define _VECTOR4_H_


#include "../basic.h"


// external

namespace de
{
    struct __align__(16) Vector3f {
        float x, y, z;
    };

    struct __align__(8) Vector2f {
        float x, y;
    };

    struct __align__(16) Vector4f {
        float x, y, z, w;
    };
}



#endif
