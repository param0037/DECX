/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _GEMM_FP16_CUH_
#define _GEMM_FP16_CUH_

#include "../../GEMM_kernel_def.cuh"
#include "../../../GEMM_utils.h"


namespace decx {
    namespace gemm {
        namespace GPUK {

            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_spec(float4* A, float4* B, float4* dst, const uint pitch_A, const uint pitch_B, const uint __iter);



            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
            __global__
                /**
                * config -> <<<dim3(h / 128, w / 128, 1), int(16 * 16), 0, S>>>
                * __same should be 16-times and dstDims should be both 128-times
                * @param pitch_A : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_B : considered float4, is the true width on device memory (>= ~.width)
                * @param pitch_dst : considered float4
                * @param __iter : __linear(in float) / 16
                */
                void cu_GEMM_fp16_ABC_spec(float4* A, float4* B, float4* C, float4* dst, const uint pitch_A,
                    const uint pitch_B, const uint __iter);


            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
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
                void cu_GEMM_fp16_anyWH_anyL(float4* A,
                    float4* B,
                    float4* dst,
                    const uint pitch_A,
                    const uint pitch_B,
                    const uint Hdst,
                    const uint HB,
                    const uint __iter);


            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
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
                void cu_GEMM_fp16_ABC_anyWH_anyL(float4* A,
                    float4* B,
                    float4* C,
                    float4* dst,
                    const uint                pitch_A,
                    const uint                pitch_B,
                    const uint                Hdst,
                    const uint                HB,
                    const uint                __iter);



            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
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
                void cu_GEMM_fp16_anyWH_specL(float4* A,
                    float4* B,
                    float4* dst,
                    const uint                pitch_A,
                    const uint                pitch_B,
                    const uint                Hdst,
                    const uint                __iter);


            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
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
                void cu_GEMM_fp16_ABC_anyWH_specL(float4* A,
                    float4* B,
                    float4* C,
                    float4* dst,
                    const uint                pitch_A,
                    const uint                pitch_B,
                    const uint                Hdst,
                    const uint                __iter);




            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
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
                void cu_GEMM_fp16_specWH_anyL(float4* A,
                    float4* B,
                    float4* dst,
                    const uint                pitch_A,
                    const uint                pitch_B,
                    const uint                HB,
                    const uint                __iter);



            // last storage (16, 16)
            // ���� / �ô� �� is the crucial, reduce memory assess by vectorization
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
                void cu_GEMM_fp16_ABC_specWH_anyL(float4* A,
                    float4* B,
                    float4* C,
                    float4* dst,
                    const uint                pitch_A,
                    const uint                pitch_B,
                    const uint                HB,
                    const uint                __iter);

        }
    }
}

#endif