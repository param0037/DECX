/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _REDUCTION_SUM_CUH_
#define _REDUCTION_SUM_CUH_

#include "../../classes/classes_util.h"
#include "../../core/utils/decx_utils_device_functions.cuh"


#define REDUCTION_BLOCK_SIZE 512

namespace decx
{
    namespace bp {
        namespace GPUK {
            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_sum_vec4_fp32(float4* A, float4* B, const size_t thr_num, const size_t dst_len);


            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_sum_vec8_fp16(float4* A, float4* B, const size_t thr_num, const size_t dst_len);



            __global__
            /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_sum_vec8_fp16_accu_fp16_output(float4* A, float4* B, const size_t thr_num, const size_t dst_len);



            __global__
                /*
            * This kernel function contains multiply-add operation
            * @param thr_num : The threads number is half of the total length
            */
            void cu_sum_vec8_fp16_accu_fp32_output(float4* A, float4* B, const size_t thr_num, const size_t dst_len);
        }
    }
}



namespace decx
{
    template <typename T, typename vec_T>
    static void _final_vec4_sum(vec_T* _in, T* res);


    static void _final_vec8_sum_fp16(half2_8* _in, float* res);
}


template <typename T, typename vec_T>
static void decx::_final_vec4_sum(vec_T* _in, T* res)
{
    T _sum;
    _sum = _in->x;           _sum += _in->y;
    _sum += _in->z;          _sum += _in->w;
    *res = _sum;
}


static void decx::_final_vec8_sum_fp16(half2_8* _in, float* res)
{
    float _ans = 0;
    _ans += __half2float(_in->x.x);         _ans += __half2float(_in->x.y);
    _ans += __half2float(_in->y.x);         _ans += __half2float(_in->y.y);
    _ans += __half2float(_in->z.x);         _ans += __half2float(_in->z.y);
    _ans += __half2float(_in->w.x);         _ans += __half2float(_in->w.y);
   *res = _ans;
}



#endif