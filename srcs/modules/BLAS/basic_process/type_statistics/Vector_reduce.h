/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _VECTOR_REDUCE_H_
#define _VECTOR_REDUCE_H_

#include "../../../classes/Vector.h"
#include "../../../core/configs/config.h"
#include "../../../classes/classes_util.h"
#ifdef _DECX_CUDA_PARTS_
#include "../../../basic_calculations/reduce/CUDA/reduce_callers.cuh"
#include "../../../classes/GPU_Vector.h"
#include "../../../core/cudaStream_management/cudaEvent_queue.h"
#endif


namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Sum(de::Vector& src, de::DecxNumber* res, const uint32_t _fp16_accu);


        _DECX_API_ de::DH Max(de::Vector& src, de::DecxNumber* res);


        _DECX_API_ de::DH Min(de::Vector& src, de::DecxNumber* res);


        _DECX_API_ de::DH Sum(de::GPU_Vector& src, de::DecxNumber* res, const uint32_t _fp16_accu);


        _DECX_API_ de::DH Max(de::GPU_Vector& src, de::DecxNumber* res);


        _DECX_API_ de::DH Min(de::GPU_Vector& src, de::DecxNumber* res);
    }
}


#endif