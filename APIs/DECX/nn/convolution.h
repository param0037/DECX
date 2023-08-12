/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT_H_
#define _FFT_H_

#include "../basic.h"
#include "../classes/Vector.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/class_utils.h"


namespace de {
    namespace cpu 
    {
        _DECX_API_ de::DH Conv2D(de::Tensor& src, de::TensorArray& kernel, de::Tensor& dst, const de::Point2D strideXY, const int conv_flag);
    }

    namespace cuda {
        _DECX_API_ de::DH Conv2D(de::GPU_Tensor& src, de::GPU_TensorArray& kernel,
            de::GPU_Tensor& dst, const de::Point2D strideXY, const int flag, const int accu_flag);
    }
}


#endif