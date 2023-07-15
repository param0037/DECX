/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _SHAPE_CAST_H_
#define _SHAPE_CAST_H_

#include "../classes/Matrix.h"
#include "../classes/GPU_Matrix.h"



namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst);
    }

    namespace cpu {
        _DECX_API_ de::DH Transpose(de::Matrix& src, de::Matrix& dst);
    }
}


#endif