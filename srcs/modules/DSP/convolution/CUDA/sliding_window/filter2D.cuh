/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FILTER2D_CUH_
#define _FILTER2D_CUH_


#include "../../../../classes/GPU_Matrix.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace de
{
    namespace dsp {
        namespace cuda {
            _DECX_API_ de::DH Filter2D(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
                const de::extend_label _extend_method);
        }
    }
}


#endif