/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _MODULE_FP32_H_
#define _MODULE_FP32_H_


#include "module_fp32_exec.h"
#include "../../classes/Vector.h"
#include "../../classes/Matrix.h"


namespace de
{
    namespace signal {
        namespace CPU {
            _DECX_API_ de::DH Module(de::Matrix& src, de::Matrix& dst);
        }
    }
}


#endif