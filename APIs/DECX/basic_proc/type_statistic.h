/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _TYPE_STATISTIC_H_
#define _TYPE_STATISTIC_H_

#include "../classes/Matrix.h"
#include "../classes/Vector.h"
#include "../classes/Tensor.h"
#include "../classes/GPU_Matrix.h"
#include "../classes/GPU_Vector.h"
#include "../classes/GPU_Tensor.h"
#include "../basic.h"

namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Min_fp32(de::Vector& src, float* res);


        _DECX_API_ de::DH Min_fp32(de::Matrix& src, float* res);


        _DECX_API_ de::DH Min_fp32(de::Tensor& src, float* res);


        _DECX_API_ de::DH Min_fp64(de::Vector& src, double* res);


        _DECX_API_ de::DH Min_fp64(de::Matrix& src, double* res);


        _DECX_API_ de::DH Min_fp64(de::Tensor& src, double* res);
    }
}


// old version

//namespace de
//{
//    namespace cuda
//    {
//        _DECX_API_ de::DH Min_fp32(de::GPU_Vector& src, float* res);
//
//
//        _DECX_API_ de::DH Min_int32(de::GPU_Vector& src, int* res);
//
//
//        _DECX_API_ de::DH Min_fp16(de::GPU_Vector& src, de::Half* res);
//
//
//        _DECX_API_ de::DH Min_fp32(de::GPU_Matrix& src, float* res);
//
//
//        _DECX_API_ de::DH Min_int32(de::GPU_Matrix& src, int* res);
//
//
//        _DECX_API_ de::DH Min_fp16(de::GPU_Matrix& src, de::Half* res);
//
//
//        _DECX_API_ de::DH Max_fp32(de::GPU_Vector& src, float* res);
//
//
//        _DECX_API_ de::DH Max_int32(de::GPU_Vector& src, int* res);
//
//
//        _DECX_API_ de::DH Max_fp16(de::GPU_Vector& src, de::Half* res);
//
//
//        _DECX_API_ de::DH Max_fp32(de::GPU_Matrix& src, float* res);
//
//
//        _DECX_API_ de::DH Max_int32(de::GPU_Matrix& src, int* res);
//
//
//        _DECX_API_ de::DH Sum_fp16(de::GPU_Matrix& src, de::Half* res);
//
//
//        _DECX_API_ de::DH Sum_fp32(de::GPU_Vector& src, float* res);
//
//
//        _DECX_API_ de::DH Sum_int32(de::GPU_Vector& src, int* res);
//
//
//        _DECX_API_ de::DH Sum_fp16(de::GPU_Vector& src, de::Half* res);
//
//
//        _DECX_API_ de::DH Sum_fp32(de::GPU_Matrix& src, float* res);
//
//
//        _DECX_API_ de::DH Sum_int32(de::GPU_Matrix& src, int* res);
//
//
//        _DECX_API_ de::DH Sum_fp16(de::GPU_Matrix& src, de::Half* res);
//    }
//}

// latest

namespace de
{
    enum REDUCE_METHOD
    {
        _REDUCE2D_FULL_ = 0,
        _REDUCE2D_H_ = 1,
        _REDUCE2D_V_ = 2,
    };


    namespace cuda
    {
        _DECX_API_ de::DH Sum(de::Vector& src, double* res);
        _DECX_API_ de::DH Sum_Async(de::Vector& src, double* res, de::DecxStream& S);


        _DECX_API_ de::DH Max(de::Vector& src, double* res);
        _DECX_API_ de::DH Max_Async(de::Vector& src, double* res, de::DecxStream& S);


        _DECX_API_ de::DH Min(de::Vector& src, double* res);
        _DECX_API_ de::DH Min_Async(de::Vector& src, double* res, de::DecxStream& S);


        _DECX_API_ de::DH Sum(de::GPU_Vector& src, double* res);
        _DECX_API_ de::DH Sum_Async(de::GPU_Vector& src, double* res, de::DecxStream& S);


        _DECX_API_ de::DH Max(de::GPU_Vector& src, double* res);
        _DECX_API_ de::DH Max_Async(de::GPU_Vector& src, double* res, de::DecxStream& S);


        _DECX_API_ de::DH Min(de::GPU_Vector& src, double* res);
        _DECX_API_ de::DH Min_Async(de::GPU_Vector& src, double* res, de::DecxStream& S);


        _DECX_API_ de::DH Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);


        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);


        _DECX_API_ de::DH Sum(de::Matrix& src, double* res);
        _DECX_API_ de::DH Max(de::Matrix& src, double* res);
        _DECX_API_ de::DH Min(de::Matrix& src, double* res);


        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, double* res);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, double* res);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, double* res);
    }
}

#endif