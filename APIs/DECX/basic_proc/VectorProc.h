/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _VECTORPROC_H_
#define _VECTORPROC_H_


#include "../classes/Vector.h"
#include "../classes/GPU_Vector.h"


namespace de
{
    enum extend_label {
        _EXTEND_NONE_ = 0,
        _EXTEND_REFLECT_ = 1,
        _EXTEND_CONSTANT_ = 2,
    };
}




namespace de {
    namespace cpu 
    {
        _DECX_API_ de::DH Constant_fp32(de::Vector& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::Vector& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::Vector& src, const double value);
    }

    namespace cuda {
        _DECX_API_ de::DH Constant_fp32(GPU_Vector& src, const float value);


        _DECX_API_ de::DH Constant_int32(GPU_Vector& src, const int value);


        _DECX_API_ de::DH Constant_fp64(GPU_Vector& src, const double value);
    }
}



namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Extend(de::Vector& src, de::Vector& dst, const uint32_t left, const uint32_t right,
            const int border_type, void* val);
    }
}


namespace de
{
    namespace gen {
        namespace cpu {
            _DECX_API_ de::DH Zeros(de::Vector& src);
        }
    }
}


#endif