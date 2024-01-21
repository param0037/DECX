/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _ZEROS_CUH_
#define _ZEROS_CUH_


#include "../../../classes/GPU_Vector.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/GPU_Tensor.h"



namespace de
{
    namespace gen {
        namespace cuda {
            _DECX_API_ de::DH Zeros(de::GPU_Vector& src);


            _DECX_API_ de::DH Zeros(de::GPU_Matrix& src);


            _DECX_API_ de::DH Zeros(de::GPU_Tensor& src);
        }
    }
}



#endif