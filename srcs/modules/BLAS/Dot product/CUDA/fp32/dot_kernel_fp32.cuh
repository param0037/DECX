/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _DOT_FP32_CUH_
#define _DOT_FP32_CUH_


#include "../../../../classes/classes_util.h"
#include "../../../basic_process/type_statistics/CUDA/reduction_sum.cuh"


namespace decx
{
    namespace dot {
        namespace GPUK {
            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_dot_vec4f_start(float4* A, float4* B, float4* dst, const size_t thr_num, const size_t dst_len);




            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_dot_vec4f(float4* A, float4* B, const size_t thr_num, const size_t dst_len);
        }
    }
}




#endif