/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _EQ_GEMM_FP16_CUH_
#define _EQ_GEMM_FP16_CUH_


#include "../../../../core/basic.h"
#include "../fp32/eq_GEMM_fp32.cuh"
#include "../../../../classes/classes_util.h"
#include "../eq_MM_device_funcs.cuh"


namespace decx 
{
    namespace conv {
        namespace GPUK {
            __global__
            /*
            * Threads distribution : int(8 x 32) = (256)
            * Matrix A has limitation on its boundaries, but Matrix B dose not
            * 
            * @param MatA_load_bounds : .x : width of matrix A(in float4) (how many float4 in a row)
            *                            .y : height of matrix A(in float4), I2C->height is 4x (how many float4 in a colume)
            * @param dst_store_bound : the width of dst, in float4
            * @param WB : width of matrix B (in float4)
            * @param __iter : how many iteration along B's width (128x)
            */
            void cu_conv_eq_mm_fp16(float4*                  A,
                                    float4*                  B,
                                    float4*                  dst,
                                    const ulong2             MatA_load_bounds,
                                    const uint               dst_store_bound,                // in float4
                                    const uint               WB,                             // in float4
                                    const uint               __iter);


            __global__
            /*
            * Threads distribution : int(8 x 32) = (256)
            * Matrix A has limitation on its boundaries, but Matrix B dose not
            * 
            * @param MatA_load_bounds : .x : width of matrix A(in float4) (how many float4 in a row)
            *                            .y : height of matrix A(in float4), I2C->height is 4x (how many float4 in a colume)
            * @param dst_store_bound : the width of dst, in float4
            * @param WB : width of matrix B (in float4)
            * @param __iter : how many iteration along B's width (128x)
            */
            void cu_conv_eq_mm_fp16_accu(float4*                  A,
                                         float4*                  B,
                                         float4*                  dst,
                                         const ulong2             MatA_load_bounds,
                                         const uint               dst_store_bound,                // in float4
                                         const uint               WB,                             // in float4
                                         const uint               __iter);
        }
    }
}



#endif