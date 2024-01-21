/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "reduction_sum.cuh"



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void decx::bp::GPUK::cu_sum_vec4_fp32(float4* A, float4* B, const size_t thr_num, const size_t dst_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE];
    float4 tmp[2];
    tmp[1] = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = tmp[1];

    if (tid < dst_len) {
        B[tid] = tmp[1];
    }

    __syncthreads();

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
void decx::bp::GPUK::cu_sum_vec8_fp16(float4* A, float4* B, const size_t thr_num, const size_t dst_len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE];
    half2_8 tmp[2];
    *((float4*)&tmp[1]) = make_float4(0, 0, 0, 0);
    shmem[threadIdx.x] = *((float4*)&tmp[1]);

    if (tid < dst_len) {
        B[tid] = *((float4*)&tmp[1]);
    }

    if (tid < thr_num) {
        *((float4*)&tmp[0]) = A[tid * 2];
        *((float4*)&tmp[1]) = A[tid * 2 + 1];
        tmp[1].x = __hadd2(tmp[0].x, tmp[1].x);
        tmp[1].y = __hadd2(tmp[0].y, tmp[1].y);
        tmp[1].z = __hadd2(tmp[0].z, tmp[1].z);
        tmp[1].w = __hadd2(tmp[0].w, tmp[1].w);

        shmem[threadIdx.x] = *((float4*)&tmp[1]);
    }

    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            *((float4*)&tmp[0]) = shmem[threadIdx.x];
            *((float4*)&tmp[1]) = shmem[threadIdx.x + _step];

            tmp[1].x = __hadd2(tmp[0].x, tmp[1].x);
            tmp[1].y = __hadd2(tmp[0].y, tmp[1].y);
            tmp[1].z = __hadd2(tmp[0].z, tmp[1].z);
            tmp[1].w = __hadd2(tmp[0].w, tmp[1].w);
        }
        __syncthreads();
        shmem[threadIdx.x] = *((float4*)&tmp[1]);
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        B[blockIdx.x] = shmem[0];
    }
#endif
}



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void decx::bp::GPUK::cu_sum_vec8_fp16_accu_fp16_output(float4* A, float4* B, const size_t thr_num, const size_t dst_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE * 2];

    half2_8 tmp[2];
    float4& float_tmp0 = *((float4*)&tmp[0]);
    float4& float_tmp1 = *((float4*)&tmp[1]);
    float4 _mid_res[2];

    _mid_res[0] = make_float4(0, 0, 0, 0);
    _mid_res[1] = make_float4(0, 0, 0, 0);

    if (tid < dst_len) {
        B[tid] = _mid_res[0];
    }

    shmem[threadIdx.x * 2] = _mid_res[0];
    shmem[threadIdx.x * 2 + 1] = _mid_res[0];

    if (tid < thr_num) {
        *((float4*)&tmp[0]) = A[tid * 2];
        *((float4*)&tmp[1]) = A[tid * 2 + 1];

        _mid_res[0].x = __fadd_rn(__half2float(tmp[0].x.x), __half2float(tmp[1].x.x));
        _mid_res[0].y = __fadd_rn(__half2float(tmp[0].x.y), __half2float(tmp[1].x.y));
        _mid_res[0].z = __fadd_rn(__half2float(tmp[0].y.x), __half2float(tmp[1].y.x));
        _mid_res[0].w = __fadd_rn(__half2float(tmp[0].y.y), __half2float(tmp[1].y.y));
        _mid_res[1].x = __fadd_rn(__half2float(tmp[0].z.x), __half2float(tmp[1].z.x));
        _mid_res[1].y = __fadd_rn(__half2float(tmp[0].z.y), __half2float(tmp[1].z.y));
        _mid_res[1].z = __fadd_rn(__half2float(tmp[0].w.x), __half2float(tmp[1].w.x));
        _mid_res[1].w = __fadd_rn(__half2float(tmp[0].w.y), __half2float(tmp[1].w.y));

        shmem[threadIdx.x * 2] = _mid_res[0];
        shmem[threadIdx.x * 2 + 1] = _mid_res[1];
    }

    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step) {
            *((float4*)&tmp[0]) = shmem[threadIdx.x * 2];
            *((float4*)&tmp[1]) = shmem[(threadIdx.x + _step) * 2];

            _mid_res[0].x = __fadd_rn(float_tmp0.x, float_tmp1.x);
            _mid_res[0].y = __fadd_rn(float_tmp0.y, float_tmp1.y);
            _mid_res[0].z = __fadd_rn(float_tmp0.z, float_tmp1.z);
            _mid_res[0].w = __fadd_rn(float_tmp0.w, float_tmp1.w);

            float_tmp0 = shmem[threadIdx.x * 2 + 1];
            float_tmp1 = shmem[(threadIdx.x + _step) * 2 + 1];

            _mid_res[1].x = __fadd_rn(float_tmp0.x, float_tmp1.x);
            _mid_res[1].y = __fadd_rn(float_tmp0.y, float_tmp1.y);
            _mid_res[1].z = __fadd_rn(float_tmp0.z, float_tmp1.z);
            _mid_res[1].w = __fadd_rn(float_tmp0.w, float_tmp1.w);
        }
        __syncthreads();
        shmem[threadIdx.x * 2] = _mid_res[0];
        shmem[threadIdx.x * 2 + 1] = _mid_res[1];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        _mid_res[0] = shmem[0];              _mid_res[1] = shmem[1];
        // transfer back to half
        tmp[0].x.x = __float2half(_mid_res[0].x);           tmp[0].x.y = __float2half(_mid_res[0].y);
        tmp[0].y.x = __float2half(_mid_res[0].z);           tmp[0].y.y = __float2half(_mid_res[0].w);
        tmp[0].z.x = __float2half(_mid_res[1].x);           tmp[0].z.y = __float2half(_mid_res[1].y);
        tmp[0].w.x = __float2half(_mid_res[1].z);           tmp[0].w.y = __float2half(_mid_res[1].w);
        B[blockIdx.x] = *((float4*)&tmp[0]);
    }
}



__global__
/*
* This kernel function contains multiply-add operation
* @param thr_num : The threads number is half of the total length
*/
void decx::bp::GPUK::cu_sum_vec8_fp16_accu_fp32_output(float4* A, float4* B, const size_t thr_num, const size_t dst_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float4 shmem[REDUCTION_BLOCK_SIZE * 2];

    half2_8 tmp[2];
    float4& float_tmp0 = *((float4*)&tmp[0]);
    float4& float_tmp1 = *((float4*)&tmp[1]);
    float4 _mid_res[2];

    _mid_res[0] = make_float4(0, 0, 0, 0);
    _mid_res[1] = make_float4(0, 0, 0, 0);

    if (tid < dst_len) {
        B[tid] = _mid_res[0];
    }

    shmem[threadIdx.x * 2] = *((float4*)&_mid_res[0]);
    shmem[threadIdx.x * 2 + 1] = *((float4*)&_mid_res[0]);

    if (tid < thr_num) {
        *((float4*)&tmp[0]) = A[tid * 2];
        *((float4*)&tmp[1]) = A[tid * 2 + 1];

        _mid_res[0].x = __fadd_rn(__half2float(tmp[0].x.x), __half2float(tmp[1].x.x));
        _mid_res[0].y = __fadd_rn(__half2float(tmp[0].x.y), __half2float(tmp[1].x.y));
        _mid_res[0].z = __fadd_rn(__half2float(tmp[0].y.x), __half2float(tmp[1].y.x));
        _mid_res[0].w = __fadd_rn(__half2float(tmp[0].y.y), __half2float(tmp[1].y.y));
        _mid_res[1].x = __fadd_rn(__half2float(tmp[0].z.x), __half2float(tmp[1].z.x));
        _mid_res[1].y = __fadd_rn(__half2float(tmp[0].z.y), __half2float(tmp[1].z.y));
        _mid_res[1].z = __fadd_rn(__half2float(tmp[0].w.x), __half2float(tmp[1].w.x));
        _mid_res[1].w = __fadd_rn(__half2float(tmp[0].w.y), __half2float(tmp[1].w.y));

        shmem[threadIdx.x * 2] = _mid_res[0];
        shmem[threadIdx.x * 2 + 1] = _mid_res[1];
    }
    __syncthreads();

    // Take tmp[2] as summing result, tmp[0], tmp[1] as tmps
    int _step = REDUCTION_BLOCK_SIZE / 2;
#pragma unroll 9
    for (int i = 0; i < 9; ++i) {
        if (threadIdx.x < _step)
        {
            float_tmp0 = shmem[threadIdx.x * 2];
            float_tmp1 = shmem[(threadIdx.x + _step) * 2];

            _mid_res[0].x = __fadd_rn(float_tmp0.x, float_tmp1.x);
            _mid_res[0].y = __fadd_rn(float_tmp0.y, float_tmp1.y);
            _mid_res[0].z = __fadd_rn(float_tmp0.z, float_tmp1.z);
            _mid_res[0].w = __fadd_rn(float_tmp0.w, float_tmp1.w);

            float_tmp0 = shmem[threadIdx.x * 2 + 1];
            float_tmp1 = shmem[(threadIdx.x + _step) * 2 + 1];

            _mid_res[1].x = __fadd_rn(float_tmp0.x, float_tmp1.x);
            _mid_res[1].y = __fadd_rn(float_tmp0.y, float_tmp1.y);
            _mid_res[1].z = __fadd_rn(float_tmp0.z, float_tmp1.z);
            _mid_res[1].w = __fadd_rn(float_tmp0.w, float_tmp1.w);
        }
        __syncthreads();
        shmem[threadIdx.x * 2] = _mid_res[0];
        shmem[threadIdx.x * 2 + 1] = _mid_res[1];
        __syncthreads();
        _step /= 2;
    }
    if (threadIdx.x == 0) {
        float_tmp0 = shmem[0];              float_tmp1 = shmem[1];
        //decx::utils::add_vec4(&float_tmp0, &float_tmp1, &_mid_res[0]);
        B[blockIdx.x] = _mid_res[0];
    }
}