/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _EDGE_DETECTORS_H_
#define _EDGE_DETECTORS_H_


#include "../../../core/basic.h"
#include "../../../classes/Matrix.h"
#include "edge_det_ops.h"
#include "../../../core/utils/fragment_arrangment.h"


namespace de
{
    namespace vis {
        enum Canny_Methods {
            DE_SOBEL = 0,
            DE_SCHARR = 1
        };

        namespace cpu {
            _DECX_API_ de::DH
                Find_Edge(de::Matrix& src, de::Matrix& dst, const float _L_threshold, const float _H_threshold, const int method);
        }
    }
}


#endif