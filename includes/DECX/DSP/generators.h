/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GENERATORS_H_
#define _GENERATORS_H_


#include "../classes/Vector.h"
#include "../classes/Matrix.h"
#include "../classes/Tensor.h"


namespace de
{
    namespace dsp {
        namespace cpu {
            _DECX_API_ void Zeros(de::Vector& src);


            _DECX_API_ void Zeros(de::Matrix& src);


            _DECX_API_ void Zeros(de::Tensor& src);


            _DECX_API_ void RandomGaussian(de::Matrix& src, const float mean, const float sigma, de::Point2D_d clipping_range,
                const uint32_t resolution, const int data_type);
        }
    }
}


#endif