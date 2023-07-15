/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_MATH_H_
#define _MATRIX_MATH_H_

#include "../math_functions_raw_API.h"
#include "../../../classes/Matrix.h"
#include "../../../core/configs/config.h"


namespace de
{
    namespace cpu 
    {
        _DECX_API_ de::DH Log10(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Log2(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Exp(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Sin(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Cos(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Tan(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Asin(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Acos(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Atan(de::Matrix& src, de::Matrix& dst);

        _DECX_API_ de::DH Sqrt(de::Matrix& src, de::Matrix& dst);
    }
}


#endif