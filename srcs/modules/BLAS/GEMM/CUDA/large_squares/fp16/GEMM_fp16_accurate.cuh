/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _GEMM_FP16_ACCURATE_CUH_
#define _GEMM_FP16_ACCURATE_CUH_

#include "../../GEMM_kernel_def.cuh"
#include "../../../GEMM_utils.h"
#include "../../GEMM_kernel_def.cuh"
//#include <mma.h>


namespace decx {
    namespace gemm {
        namespace GPUK {
            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_spec_accu(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_ABC_spec_accu(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_anyWH_anyL_accu(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint HB, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_ABC_anyWH_anyL_accu(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint HB, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_specWH_anyL_accu(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint HB, const uint __iter);




            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_ABC_specWH_anyL_accu(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint HB, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_anyWH_specL_accu(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_ABC_anyWH_specL_accu(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint __iter);

        }
    }
}

#endif