/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _TRANSPOSE_H_
#define _TRANSPOSE_H_


#include "transpose_exec.h"
#include "../../../../classes/Matrix.h"


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Transpose(de::Matrix& src, de::Matrix& dst);
    }
}


#endif