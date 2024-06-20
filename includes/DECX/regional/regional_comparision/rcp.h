/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _RCP_H_
#define _RCP_H_

#include "../../classes/Matrix.h"
#include "rcp_flags.h"


namespace de
{
    namespace rcp {
        enum _RCP_FLAGS_ {
            RCP_SQDIFF = 0,
            RCP_SQDIFF_NORMAL = 1,
            RCP_CCOEFF = 2,
            RCP_CCOEFF_NORMAL = 3
        };
    }

    namespace cpu
    {
        _DECX_API_ de::DH Regional_Comparision(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int conv_flag, const int calc_flag);
    }
}


#endif