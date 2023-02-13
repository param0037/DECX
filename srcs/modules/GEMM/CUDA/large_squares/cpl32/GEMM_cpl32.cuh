/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _GEMM_CPL32_CUH_
#define _GEMM_CPL32_CUH_

#include "../../GEMM_kernel_def.cuh"
#include "../../../GEMM_utils.h"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"
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
                void cu_GEMM_cpl32_spec(float4* A, float4* B, float4* dst,
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
                void cu_GEMM_cpl32_ABC_spec(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint __iter);





            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param bounds : ~.x : width (in float4); ~.y : height
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_cpl32_anyWH_specL(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint __iter);




            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param HB : lenght of linear region(_B.height) (in float)
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_cpl32_ABC_anyWH_specL(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint __iter);





            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param HB : lenght of linear region(_B.height) (in float)
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_cpl32_anyWH_anyL(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint HB, const uint __iter);



            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param HB : lenght of linear region(_B.height) (in float)
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_cpl32_ABC_anyWH_anyL(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint Hdst, const uint HB, const uint __iter);



            // last storage (16, 16)
            // 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param HB : lenght of linear region(_B.height) (in float)
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_cpl32_specWH_anyL(float4* A, float4* B, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint HB, const uint __iter);



            // last storage (16, 16)
            // 计算 / 访存 比 is the crucial, reduce memory assess by vectorization
            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param HB : lenght of linear region(_B.height) (in float)
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_cpl32_ABC_specWH_anyL(float4* A, float4* B, float4* C, float4* dst,
                    const uint pitch_A, const uint pitch_B, const uint HB, const uint __iter);

        }
    }
}


#endif