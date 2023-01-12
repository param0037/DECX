/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DOT_H_
#define _DOT_H_

#include "../classes/Vector.h"
#include "../classes/Matrix.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Tensor.h"



namespace de
{
    enum Dot_Fp16_Accuracy_Levels
    {
        Dot_Fp16_Accurate_L0 = 0,
        Dot_Fp16_Accurate_L1 = 1,
        Dot_Fp16_Accurate_L2 = 2,
    };
}


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Dot(de::Vector& A, de::Vector& B, float* res);


        _DECX_API_ de::DH Dot(de::Matrix& A, de::Matrix& B, float* res);


        _DECX_API_ de::DH Dot(de::Tensor& A, de::Tensor& B, float* res);
    }


    namespace cuda
    {
        _DECX_API_ de::DH Dot_fp32(de::GPU_Vector& A, de::GPU_Vector& B, float* res);


        _DECX_API_ de::DH Dot_fp32(de::GPU_Matrix& A, de::GPU_Matrix& B, float* res);


        _DECX_API_ de::DH Dot_fp32(de::GPU_Tensor& A, de::GPU_Tensor& B, float* res);


        _DECX_API_ de::DH Dot_fp16(de::GPU_Vector& A, de::GPU_Vector& B, de::Half* res, const int accu_flag);


        _DECX_API_ de::DH Dot_fp16(de::GPU_Matrix& A, de::GPU_Matrix& B, de::Half* res, const int accu_flag);


        _DECX_API_ de::DH Dot_fp16(de::GPU_Tensor& A, de::GPU_Tensor& B, de::Half* res, const int accu_flag);
    }
}


#endif