/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _RANDOM_H_
#define _RANDOM_H_


#include "distributions_exec.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/Vector.h"


namespace de
{
    namespace dsp {
        namespace cpu {
            _DECX_API_ void RandomGaussian(de::Matrix& src, const float mean, const float sigma, de::Point2D_d clipping_range, 
                const uint32_t resolution, const int data_type);
        }
    }
}


#endif