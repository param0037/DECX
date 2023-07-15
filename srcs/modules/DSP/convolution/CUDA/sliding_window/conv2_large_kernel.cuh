/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CONV2_LARGE_KERNEL_CUH_
#define _CONV2_LARGE_KERNEL_CUH_

#include "../../../core/basic.h"
// to include the macro definition : REDUCTION_VEC4_BLOCK
#include "../../../basic_process/type_statistics/reduction_sum.cuh"


__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_sConv2_LK_start(float4* src, float4* kernel, float4* dst, const size_t thr_num)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE];
    float4 tmp[3];
    tmp[2] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[2];

    if (tid < thr_num) {
        tmp[0] = src[2 * tid];
        tmp[1] = kernel[2 * tid];
        tmp[2].x = __fmul_rn(tmp[0].x, tmp[1].x);
        tmp[2].y = __fmul_rn(tmp[0].y, tmp[1].y);
        tmp[2].z = __fmul_rn(tmp[0].z, tmp[1].z);
        tmp[2].w = __fmul_rn(tmp[0].w, tmp[1].w);

        tmp[0] = src[2 * tid + 1];
        tmp[1] = kernel[2 * tid + 1];
        tmp[2].x = fmaf(tmp[0].x, tmp[1].x, tmp[2].x);
        tmp[2].y = fmaf(tmp[0].y, tmp[1].y, tmp[2].y);
        tmp[2].z = fmaf(tmp[0].z, tmp[1].z, tmp[2].z);
        tmp[2].w = fmaf(tmp[0].w, tmp[1].w, tmp[2].w);

        shmem[threadIdx.x] = tmp[2];
    }
    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[2].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[2].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[2].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[2].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[2];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        dst[blockIdx.x] = shmem[0];
    }
}



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_sConv2_LK_once(float4* src, float4* kernel, float* dst, const size_t thr_num, const size_t dst_dex)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE];
    float4 tmp[3];
    tmp[2] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[2];

    if (tid < thr_num) {
        tmp[0] = src[2 * tid];
        tmp[1] = kernel[2 * tid];
        tmp[2].x = __fmul_rn(tmp[0].x, tmp[1].x);
        tmp[2].y = __fmul_rn(tmp[0].y, tmp[1].y);
        tmp[2].z = __fmul_rn(tmp[0].z, tmp[1].z);
        tmp[2].w = __fmul_rn(tmp[0].w, tmp[1].w);

        tmp[0] = src[2 * tid + 1];
        tmp[1] = kernel[2 * tid + 1];
        tmp[2].x = fmaf(tmp[0].x, tmp[1].x, tmp[2].x);
        tmp[2].y = fmaf(tmp[0].y, tmp[1].y, tmp[2].y);
        tmp[2].z = fmaf(tmp[0].z, tmp[1].z, tmp[2].z);
        tmp[2].w = fmaf(tmp[0].w, tmp[1].w, tmp[2].w);

        shmem[threadIdx.x] = tmp[2];
    }
    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[2].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[2].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[2].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[2].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[2];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        tmp[0] = shmem[0];
        tmp[0].x = __fadd_rn(tmp[0].y, tmp[0].x);
        tmp[0].x = __fadd_rn(tmp[0].z, tmp[0].x);
        tmp[0].x = __fadd_rn(tmp[0].w, tmp[0].x);
        dst[dst_dex] = tmp[0].x;
    }
}



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_sConv2_LK(float4* A, float4* B, const size_t thr_num)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE];
    float4 tmp[2];
    tmp[1] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[1];

    if (tid < thr_num) {
        tmp[0] = A[tid * 2];
        tmp[1] = A[tid * 2 + 1];
        tmp[1].x = __fadd_rn(tmp[0].x, tmp[1].x);
        tmp[1].y = __fadd_rn(tmp[0].y, tmp[1].y);
        tmp[1].z = __fadd_rn(tmp[0].z, tmp[1].z);
        tmp[1].w = __fadd_rn(tmp[0].w, tmp[1].w);

        shmem[threadIdx.x] = tmp[1];
    }

    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[1].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[1].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[1].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[1].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[1];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        B[blockIdx.x] = shmem[0];
    }
}



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void cu_sConv2_LK_end(float4* A, float* B, const size_t thr_num, const size_t dst_dex)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE];
    float4 tmp[2];
    tmp[1] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[1];

    if (tid < thr_num) {
        tmp[0] = A[tid * 2];
        tmp[1] = A[tid * 2 + 1];
        tmp[1].x = __fadd_rn(tmp[0].x, tmp[1].x);
        tmp[1].y = __fadd_rn(tmp[0].y, tmp[1].y);
        tmp[1].z = __fadd_rn(tmp[0].z, tmp[1].z);
        tmp[1].w = __fadd_rn(tmp[0].w, tmp[1].w);

        shmem[threadIdx.x] = tmp[1];
    }

    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            tmp[0] = shmem[threadIdx.x];
            tmp[1] = shmem[threadIdx.x + _step];

            tmp[1].x = __fadd_rn(tmp[0].x, tmp[1].x);
            tmp[1].y = __fadd_rn(tmp[0].y, tmp[1].y);
            tmp[1].z = __fadd_rn(tmp[0].z, tmp[1].z);
            tmp[1].w = __fadd_rn(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = tmp[1];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        tmp[0] = shmem[0];
        tmp[0].x = __fadd_rn(tmp[0].y, tmp[0].x);
        tmp[0].x = __fadd_rn(tmp[0].z, tmp[0].x);
        tmp[0].x = __fadd_rn(tmp[0].w, tmp[0].x);
        B[dst_dex] = tmp[0].x;
    }
}


#endif