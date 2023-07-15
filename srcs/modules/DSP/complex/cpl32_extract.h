/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MODULE_FP32_H_
#define _MODULE_FP32_H_


#include "cpl32_extract_exec.h"
#include "../../classes/Vector.h"
#include "../../classes/Matrix.h"


namespace de
{
    namespace signal {
        namespace cpu {
            _DECX_API_ de::DH Module(de::Matrix& src, de::Matrix& dst);


            _DECX_API_ de::DH Angle(de::Matrix& src, de::Matrix& dst);
        }
    }
}


#endif