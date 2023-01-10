/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _DOT_FP16_CUH_
#define _DOT_FP16_CUH_


#include "../../../classes/classes_util.h"
#include "../../../basic_process/type_statistics/reduction_sum.cuh"


namespace decx
{
    namespace dot {
        namespace GPUK {

            __global__
                /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_dot_vec8_fp16_start(const float4* A, const float4* B, float4* dst, const size_t thr_num, const size_t dst_len);



            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_dot_vec8h_start_accu_fp32_output(const float4* A, const float4* B, float4* dst, const size_t thr_num, const size_t dst_len);




            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_dot_vec8h_start_accu_fp16_output(const float4* A, const float4* B, float4* dst, const size_t thr_num, const size_t dst_len);

        }
    }
}




#endif