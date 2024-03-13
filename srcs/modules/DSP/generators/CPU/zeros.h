/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _ZEROS_H_
#define _ZEROS_H_


#include "../../../classes/Vector.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/Tensor.h"


namespace de
{
    namespace gen {
        namespace cpu {
            _DECX_API_ de::DH Zeros(de::Vector& src);


            _DECX_API_ de::DH Zeros(de::Matrix& src);


            _DECX_API_ de::DH Zeros(de::Tensor& src);
        }
    }
}


#endif