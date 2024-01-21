/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_REDUCE_H_
#define _MATRIX_REDUCE_H_

#include "../../../classes/Matrix.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../core/configs/config.h"
#include "../../../classes/classes_util.h"
#include "../../../../Async Engine/Async_task_threadpool/Async_Engine.h"
#include "../../../../Async Engine/DecxStream/DecxStream.h"
#include "../../../classes/DecxNumber.h"
#ifdef _DECX_CUDA_PARTS_
#include "../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../classes/GPU_Matrix.h"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#endif


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
        // 1-way
        _DECX_API_ de::DH Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);

        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode);

        // Full
        _DECX_API_ de::DH Sum(de::Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::Matrix& src, de::DecxNumber& res);
        _DECX_API_ de::DH Min(de::Matrix& src, de::DecxNumber& res);

        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, de::DecxNumber& res);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, de::DecxNumber& res);


        // Asynchronous APIs
        _DECX_API_ de::DH Sum_Async(de::Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu, de::DecxStream& S);
        _DECX_API_ de::DH Max_Async(de::Matrix& src, de::DecxNumber& res, de::DecxStream& S);
        _DECX_API_ de::DH Min_Async(de::Matrix& src, de::DecxNumber& res, de::DecxStream& S);

        _DECX_API_ de::DH Sum_Async(de::GPU_Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu, de::DecxStream& S);
        _DECX_API_ de::DH Max_Async(de::GPU_Matrix& src, de::DecxNumber& res, de::DecxStream& S);
        _DECX_API_ de::DH Min_Async(de::GPU_Matrix& src, de::DecxNumber& res, de::DecxStream& S);

        _DECX_API_ de::DH Sum_Async(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu, de::DecxStream& S);
        _DECX_API_ de::DH Max_Async(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, de::DecxStream& S);
        _DECX_API_ de::DH Min_Async(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, de::DecxStream& S);

        _DECX_API_ de::DH Sum_Async(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu, de::DecxStream& S);
        _DECX_API_ de::DH Max_Async(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, de::DecxStream& S);
        _DECX_API_ de::DH Min_Async(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, de::DecxStream& S);
    }
}


#endif