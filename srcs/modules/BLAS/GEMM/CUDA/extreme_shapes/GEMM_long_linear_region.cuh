/**
*    ---------------------------------------------------------------------
*    Author : Wayne anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#ifndef _GEMM_LONG_LINEAR_REGION_H_
#define _GEMM_LONG_LINEAR_REGION_H_

#include "../../../core/basic.h"

/*
* This file is mainly aimed to solve the general matrix multiplication in which linear region
* and width of matrix B are much longer than the another dimension
* 
*                                    [                    ]
*                                     [                    ]
*                                     [                    ]
*                                     [        B            ]
*                     linear region    [                    ] 
*                                     [                    ]
*                                     [                    ]
*                                     [___________________]
*   <---- linear region ---->
*    [                        ]        [                    ]
*    [            A            ]        [        Dst            ]
*    [_______________________]        [___________________]
*/


#define _GEMM_REG_4X1_(reg_sum, reg_1, reg_frag) {    \
    reg_sum.x = fmaf(reg_1.x, reg_frag[0].x, reg_sum.x);    \
    reg_sum.x = fmaf(reg_1.y, reg_frag[1].x, reg_sum.x);    \
    reg_sum.x = fmaf(reg_1.z, reg_frag[2].x, reg_sum.x);    \
    reg_sum.x = fmaf(reg_1.w, reg_frag[3].x, reg_sum.x);    \
                                                            \
    reg_sum.y = fmaf(reg_1.x, reg_frag[0].y, reg_sum.y);    \
    reg_sum.y = fmaf(reg_1.y, reg_frag[1].y, reg_sum.y);    \
    reg_sum.y = fmaf(reg_1.z, reg_frag[2].y, reg_sum.y);    \
    reg_sum.y = fmaf(reg_1.w, reg_frag[3].y, reg_sum.y);    \
                                                            \
    reg_sum.z = fmaf(reg_1.x, reg_frag[0].z, reg_sum.z);    \
    reg_sum.z = fmaf(reg_1.y, reg_frag[1].z, reg_sum.z);    \
    reg_sum.z = fmaf(reg_1.z, reg_frag[2].z, reg_sum.z);    \
    reg_sum.z = fmaf(reg_1.w, reg_frag[3].z, reg_sum.z);    \
                                                            \
    reg_sum.w = fmaf(reg_1.x, reg_frag[0].w, reg_sum.w);    \
    reg_sum.w = fmaf(reg_1.y, reg_frag[1].w, reg_sum.w);    \
    reg_sum.w = fmaf(reg_1.z, reg_frag[2].w, reg_sum.w);    \
    reg_sum.w = fmaf(reg_1.w, reg_frag[3].w, reg_sum.w);    \
}


__global__
/**
 * @brief  config -> <<<dim3(height_A / 16, pitch_B / 16), int(16 * 16), 0, S>>>
 *      There will be 16 x 16 threads in each block, (256 total). 
 * When loading matrix A, there is a block of 16 x 128 loaded into shared, 
 * the thread distribution is 16 x 16, each thread loads 1 float4
 * When loading matrix B, there is a block of 128 x 16 (transposed of the region above).
 * The thread distribution is 32 x 8, and load 2 float4
 * 
 * @param A pointer of Matrix A
 * @param B pointer of Matrix B
 * @param dst pointer of Matrix dst
 * @param pitch_A In float4, required 64x
 * @param pitch_B In float4, required 64x
 * @param : height_A required 16x
 */
void cu_GEMM_LongLrWB_fp32(float4*                A, 
                           float4*                B, 
                           float4*                dst,
                           //const uint2            B_
                           const uint            pitch_A,
                           const uint            pitch_B,
                           const uint            __iter)
{
    uint tidx_A = (threadIdx.x / 16) + blockIdx.x * blockDim.x, 
         tidy_A = (threadIdx.x % 16) + blockIdx.y * blockDim.y,
         tidx_B = (threadIdx.x / 8) + blockIdx.x * blockDim.x,
         tidy_B = (threadIdx.x % 8) + blockIdx.y * blockDim.y;

    size_t glo_dex_A = (size_t)tidx_A * (size_t)pitch_A + (size_t)tidy_A;
    size_t glo_dex_B = (size_t)tidx_B * (size_t)pitch_B + (size_t)tidy_B * 2;

    __shared__ float4 shmemA[16][64 / 4 + 1];        // shmemA float[16][64]
    __shared__ float4 shmemB[32 * 2][64 / 4 + 1];    // shmemB float[64][64]

    float4 reg_1, reg_frag[4], reg_sum = make_float4(0, 0, 0, 0);

    // start iteration along the linear region
    for (uint i = 0; i < __iter; ++i) {
        // load from A
        reg_1 = A[glo_dex_A];
        shmemA[threadIdx.x / 16][threadIdx.x % 16] = reg_1;

        // load_from_B, first time
        reg_1 = B[glo_dex_B];
        reg_frag[0] = B[glo_dex_B + 1];
        shmemB[threadIdx.x / 8][(threadIdx.x % 8) * 2] = reg_1;
        shmemB[threadIdx.x / 8][(threadIdx.x % 8) * 2 + 1] = reg_frag[0];

        // load_from_B, second time
        glo_dex_B += (size_t)pitch_B * 32;
        reg_1 = B[glo_dex_B];
        reg_frag[0] = B[glo_dex_B + 1];
        shmemB[threadIdx.x / 8 + 32][(threadIdx.x % 8) * 2] = reg_1;
        shmemB[threadIdx.x / 8 + 32][(threadIdx.x % 8) * 2 + 1] = reg_frag[0];

        __syncthreads();

#pragma unroll 16
        for (uint z = 0; z < 16; ++z) {
            reg_1 = shmemA[threadIdx.x / 16][z];
            reg_frag[0] = shmemB[z * 4][threadIdx.x % 16];
            reg_frag[1] = shmemB[z * 4 + 1][threadIdx.x % 16];
            reg_frag[2] = shmemB[z * 4 + 2][threadIdx.x % 16];
            reg_frag[3] = shmemB[z * 4 + 3][threadIdx.x % 16];

            _GEMM_REG_4X1_(reg_sum, reg_1, reg_frag);
        }
        glo_dex_A += 16;

        __syncthreads();
    }
    glo_dex_A = (size_t)tidx_A * (size_t)pitch_B + (size_t)tidy_A;
    dst[glo_dex_A] = reg_sum;
}


#endif