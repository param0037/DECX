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
#include "../../../classes/Vector.h"
#include "../../../core/configs/config.h"
#include "../../../classes/classes_util.h"
#include "../../../../Async Engine/Async_task_threadpool/Async_Engine.h"
#include "../../../../Async Engine/DecxStream/DecxStream.h"
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


    namespace cuda {
        _DECX_API_ de::DH Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);
        _DECX_API_ de::DH Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode);


        _DECX_API_ de::DH Sum(de::Matrix& src, double* res);
        _DECX_API_ de::DH Max(de::Matrix& src, double* res);
        _DECX_API_ de::DH Min(de::Matrix& src, double* res);


        _DECX_API_ de::DH Sum(de::GPU_Matrix& src, double* res);
        _DECX_API_ de::DH Max(de::GPU_Matrix& src, double* res);
        _DECX_API_ de::DH Min(de::GPU_Matrix& src, double* res);
    }
}


#endif