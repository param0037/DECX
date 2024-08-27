/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MATHEMATIC_H_
#define _MATHEMATIC_H_

#include "../classes/Matrix.h"
#include "../classes/Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Tensor.h"
#include "../classes/class_utils.h"


namespace de {
    enum SCAN_MODE
    {
        SCAN_MODE_INCLUSIVE  = 0,
        SCAN_MODE_EXCLUSIVE  = 1,
        SCAN_MODE_HORIZONTAL = 2,
        SCAN_MODE_VERTICAL   = 3,
        SCAN_MODE_FULL       = 4
    };
}


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Clip(de::Matrix& src, de::Matrix& dst, const de::Point2D_d range);


        _DECX_API_ de::DH Log10(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Log2(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Exp(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Sin(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Cos(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Tan(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Asin(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Acos(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Atan(de::Matrix& src, de::Matrix& dst);
    }
}


namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Integral(de::Vector& src, de::Vector& dst, const int scan_mode);


        _DECX_API_ de::DH Integral(de::GPU_Vector& src, de::GPU_Vector& dst, const int scan_mode);


        _DECX_API_ de::DH Integral(de::Matrix& src, de::Matrix& dst, const int scan2D_mode, const int scan_calc_mode);


        _DECX_API_ de::DH Integral(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int scan2D_mode, const int scan_calc_mode);
    }
}


namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Sum(de::Vector& src, de::Number* res, const uint32_t _fp16_accu);

        _DECX_API_ de::DH Max(de::Vector& src, de::Number* res);

        _DECX_API_ de::DH Min(de::Vector& src, de::Number* res);


        _DECX_API_ de::DH Sum(de::GPU_Vector& src, de::Number* res, const uint32_t _fp16_accu);

        _DECX_API_ de::DH Max(de::GPU_Vector& src, de::Number* res);


        _DECX_API_ de::DH Min(de::GPU_Vector& src, de::Number* res);


        // 1-way
        _DECX_API_ de::DH Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);

        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);

        // Full
        _DECX_API_ de::DH Sum(de::Matrix& src, de::Number& res, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::Matrix& src, de::Number& res);
        _DECX_API_ de::DH Min(de::Matrix& src, de::Number& res);

        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::Number& res, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::Number& res);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::Number& res);
    }
}



#endif
