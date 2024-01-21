/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MATRIXPROC_H_
#define _MATRIXPROC_H_


#include "VectorProc.h"
#include "../classes/Matrix.h"
#include "../classes/GPU_Matrix.h"



namespace de {
    namespace cpu 
    {
        _DECX_API_ de::DH Constant_fp32(de::Matrix& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::Matrix& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::Matrix& src, const double value);
    }
}



namespace de {
    namespace cuda 
    {
        _DECX_API_ de::DH Constant_fp32(de::GPU_Matrix& src, const float value);


        _DECX_API_ de::DH Constant_int32(de::GPU_Matrix& src, const int value);


        _DECX_API_ de::DH Constant_fp64(de::GPU_Matrix& src, const double value);
    }
}



namespace de
{
    namespace cpu 
    {
        _DECX_API_ de::DH Extend(de::Matrix& src, de::Matrix& dst, const uint32_t left, const uint32_t right,
            const uint32_t top, const uint32_t bottom, const int border_type, void* val);
    }
}


namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Transpose(de::GPU_Matrix& src, de::GPU_Matrix& dst);
    }

    namespace cpu {
        _DECX_API_ de::DH Transpose(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH Histogram(de::Matrix& src, de::Vector& dst);
    }
}


#endif