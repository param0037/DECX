/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CPU_MAXIMUM_H_
#define _CPU_MAXIMUM_H_

#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/Tensor.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/classes_util.h"

#include "cmp_exec.h"


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Max(de::Matrix& src, void* res);


        _DECX_API_ de::DH Max(de::Vector& src, void* res);


        _DECX_API_ de::DH Min(de::Matrix& src, void* res);


        _DECX_API_ de::DH Min(de::Vector& src, void* res);
    }
}



#endif