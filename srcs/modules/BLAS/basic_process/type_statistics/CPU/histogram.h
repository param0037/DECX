/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_


#include "../../../../core/basic.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/Vector.h"


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Histogram(de::Matrix& src, de::Vector& dst);
    }
}


#endif